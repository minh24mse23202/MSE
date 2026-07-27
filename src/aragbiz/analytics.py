from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple


USAGE_SCOPES = {"chat", "evaluation", "knowledge", "system"}
FEEDBACK_RATINGS = {"up", "down"}
UNAVAILABLE_KNOWLEDGE_BASE_ID = "__unavailable__"
UNAVAILABLE_KNOWLEDGE_BASE_LABEL = "Deleted or unavailable"
KNOWLEDGE_PURPOSE_PREFIXES = (
    "knowledge_",
    "embedding",
    "document_",
    "index_",
    "ingestion",
    "reindex",
)


class AnalyticsError(ValueError):
    """Raised when analytics input or persisted attribution is invalid."""


@dataclass(frozen=True)
class AnalyticsFilters:
    from_at: str
    to_at: str
    scope: str = ""
    deployment_id: str = ""
    knowledge_base_id: str = ""
    chat_configuration_id: str = ""
    purpose: str = ""
    status: str = ""
    user_id: str = ""
    rating: str = ""
    query: str = ""
    page: int = 1
    page_size: int = 25

    def normalized(self) -> "AnalyticsFilters":
        start = _parse_utc(self.from_at, "from")
        end = _parse_utc(self.to_at, "to")
        if end <= start:
            raise AnalyticsError("'to' must be later than 'from'.")
        scope = self.scope.strip().lower()
        if scope and scope not in USAGE_SCOPES:
            raise AnalyticsError(f"Unknown analytics scope: {self.scope}")
        rating = self.rating.strip().lower()
        if rating and rating not in FEEDBACK_RATINGS:
            raise AnalyticsError("Feedback rating must be 'up' or 'down'.")
        return AnalyticsFilters(
            from_at=start.isoformat(),
            to_at=end.isoformat(),
            scope=scope,
            deployment_id=self.deployment_id.strip(),
            knowledge_base_id=self.knowledge_base_id.strip(),
            chat_configuration_id=self.chat_configuration_id.strip(),
            purpose=self.purpose.strip(),
            status=self.status.strip(),
            user_id=self.user_id.strip(),
            rating=rating,
            query=self.query.strip(),
            page=max(1, int(self.page)),
            page_size=max(1, min(int(self.page_size), 100)),
        )


class AnalyticsRepository(Protocol):
    def initialize(self) -> None: ...

    def overview(self, filters: AnalyticsFilters) -> Dict[str, Any]: ...

    def usage_trend(self, filters: AnalyticsFilters) -> List[Dict[str, Any]]: ...

    def usage_breakdowns(self, filters: AnalyticsFilters, metric: str) -> Dict[str, List[Dict[str, Any]]]: ...

    def usage_events(self, filters: AnalyticsFilters) -> Dict[str, Any]: ...

    def feedback(self, filters: AnalyticsFilters) -> Dict[str, Any]: ...

    def filter_options(self, filters: AnalyticsFilters) -> Dict[str, Any]: ...

    def upsert_feedback(
        self,
        *,
        assistant_message_id: str,
        version_number: int,
        user_id: str,
        user_name: str,
        user_email: str,
        rating: str,
        comment: Optional[str],
    ) -> Dict[str, Any]: ...

    def list_message_feedback(self, message_ids: Sequence[str], user_id: str) -> List[Dict[str, Any]]: ...


class AnalyticsService:
    def __init__(self, repository: AnalyticsRepository):
        self.repository = repository
        self.repository.initialize()

    def overview(self, filters: AnalyticsFilters) -> Dict[str, Any]:
        return self.repository.overview(filters.normalized())

    def usage_trend(self, filters: AnalyticsFilters) -> List[Dict[str, Any]]:
        return self.repository.usage_trend(filters.normalized())

    def usage_breakdowns(self, filters: AnalyticsFilters, metric: str = "tokens") -> Dict[str, List[Dict[str, Any]]]:
        normalized_metric = metric.strip().lower()
        if normalized_metric not in {"tokens", "cost"}:
            raise AnalyticsError("Breakdown metric must be 'tokens' or 'cost'.")
        return self.repository.usage_breakdowns(filters.normalized(), normalized_metric)

    def usage_events(self, filters: AnalyticsFilters) -> Dict[str, Any]:
        return self.repository.usage_events(filters.normalized())

    def feedback(self, filters: AnalyticsFilters) -> Dict[str, Any]:
        return self.repository.feedback(filters.normalized())

    def filter_options(self, filters: AnalyticsFilters) -> Dict[str, Any]:
        return self.repository.filter_options(filters.normalized())

    def upsert_feedback(
        self,
        *,
        assistant_message_id: str,
        version_number: int,
        user_id: str,
        user_name: str,
        user_email: str,
        rating: str,
        comment: Optional[str] = None,
    ) -> Dict[str, Any]:
        normalized_rating = rating.strip().lower()
        if normalized_rating not in FEEDBACK_RATINGS:
            raise AnalyticsError("Feedback rating must be 'up' or 'down'.")
        if not assistant_message_id.strip():
            raise AnalyticsError("Assistant message ID is required.")
        if version_number < 1:
            raise AnalyticsError("Answer version number must be at least 1.")
        if comment is not None and len(comment) > 2000:
            raise AnalyticsError("Feedback comment cannot exceed 2,000 characters.")
        return self.repository.upsert_feedback(
            assistant_message_id=assistant_message_id,
            version_number=version_number,
            user_id=user_id,
            user_name=user_name,
            user_email=user_email,
            rating=normalized_rating,
            comment=comment.strip() if comment is not None else None,
        )

    def list_message_feedback(self, message_ids: Sequence[str], user_id: str) -> List[Dict[str, Any]]:
        normalized_ids = [str(message_id).strip() for message_id in message_ids if str(message_id).strip()]
        if not normalized_ids or not user_id.strip():
            return []
        return self.repository.list_message_feedback(normalized_ids, user_id.strip())


class PostgresAnalyticsRepository:
    def __init__(self, database_url: str):
        try:
            from sqlalchemy import create_engine  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise AnalyticsError("Install the api extra to use PostgreSQL analytics.") from exc
        self.engine = create_engine(database_url, future=True)

    def initialize(self) -> None:
        ddl = """
        ALTER TABLE IF EXISTS chat_conversations ADD COLUMN IF NOT EXISTS owner_user_id TEXT;
        ALTER TABLE IF EXISTS chat_messages ADD COLUMN IF NOT EXISTS user_id TEXT;
        ALTER TABLE IF EXISTS model_usage_events ADD COLUMN IF NOT EXISTS chat_configuration_id TEXT;
        CREATE TABLE IF NOT EXISTS chat_feedback (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            conversation_id TEXT,
            user_message_id TEXT,
            assistant_message_id TEXT NOT NULL,
            message_version_id TEXT,
            version_number INTEGER NOT NULL DEFAULT 1,
            knowledge_base_id TEXT,
            chat_configuration_id TEXT,
            trace_id TEXT,
            rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
            comment TEXT NOT NULL DEFAULT '',
            question_snapshot TEXT NOT NULL DEFAULT '',
            answer_snapshot TEXT NOT NULL DEFAULT '',
            knowledge_base_name_snapshot TEXT NOT NULL DEFAULT '',
            configuration_name_snapshot TEXT NOT NULL DEFAULT '',
            configuration_code_snapshot TEXT NOT NULL DEFAULT '',
            user_name_snapshot TEXT NOT NULL DEFAULT '',
            user_email_snapshot TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (user_id, assistant_message_id, version_number)
        );
        CREATE INDEX IF NOT EXISTS idx_model_usage_created_at ON model_usage_events(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_model_usage_kb_created ON model_usage_events(knowledge_base_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_model_usage_config_created ON model_usage_events(chat_configuration_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_chat_messages_activity ON chat_messages(created_at DESC, role, status);
        CREATE INDEX IF NOT EXISTS idx_chat_messages_user ON chat_messages(user_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_chat_feedback_updated ON chat_feedback(updated_at DESC);
        """
        with self.engine.begin() as connection:
            for statement in [item.strip() for item in ddl.split(";") if item.strip()]:
                connection.exec_driver_sql(statement)

    def overview(self, filters: AnalyticsFilters) -> Dict[str, Any]:
        from sqlalchemy import text

        usage_where, usage_params = self._usage_where(filters)
        with self.engine.begin() as connection:
            usage = connection.execute(
                text(
                    f"""
                    WITH enriched AS ({self._usage_cte()})
                    SELECT
                        COUNT(*) AS calls,
                        COUNT(*) FILTER (WHERE status = 'completed') AS completed_calls,
                        COUNT(*) FILTER (WHERE status <> 'completed') AS failed_calls,
                        COALESCE(SUM(input_tokens), 0) AS input_tokens,
                        COALESCE(SUM(output_tokens), 0) AS output_tokens,
                        COALESCE(SUM(total_tokens), 0) AS total_tokens,
                        COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost_usd,
                        COALESCE(AVG(latency_ms), 0) AS average_latency_ms
                    FROM enriched
                    {usage_where}
                    """
                ),
                usage_params,
            ).mappings().one()
            engagement = self._engagement_overview(connection, filters)
        return {
            "usage": {
                "calls": int(usage["calls"] or 0),
                "completed_calls": int(usage["completed_calls"] or 0),
                "failed_calls": int(usage["failed_calls"] or 0),
                "input_tokens": int(usage["input_tokens"] or 0),
                "output_tokens": int(usage["output_tokens"] or 0),
                "total_tokens": int(usage["total_tokens"] or 0),
                "estimated_cost_usd": round(float(usage["estimated_cost_usd"] or 0), 8),
                "average_latency_ms": round(float(usage["average_latency_ms"] or 0), 3),
            },
            "engagement": engagement,
            "from": filters.from_at,
            "to": filters.to_at,
        }

    def usage_trend(self, filters: AnalyticsFilters) -> List[Dict[str, Any]]:
        from sqlalchemy import text

        where, params = self._usage_where(filters)
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    f"""
                    WITH enriched AS ({self._usage_cte()})
                    SELECT
                        LEFT(created_at, 10) AS day,
                        COALESCE(SUM(input_tokens), 0) AS input_tokens,
                        COALESCE(SUM(output_tokens), 0) AS output_tokens,
                        COALESCE(SUM(total_tokens), 0) AS total_tokens,
                        COALESCE(SUM(estimated_cost_usd), 0) AS cost_usd,
                        COUNT(*) AS calls,
                        COUNT(*) FILTER (WHERE status <> 'completed') AS failures
                    FROM enriched
                    {where}
                    GROUP BY LEFT(created_at, 10)
                    ORDER BY day
                    """
                ),
                params,
            ).mappings()
            return [
                {
                    "day": row["day"],
                    "input_tokens": int(row["input_tokens"] or 0),
                    "output_tokens": int(row["output_tokens"] or 0),
                    "total_tokens": int(row["total_tokens"] or 0),
                    "cost_usd": round(float(row["cost_usd"] or 0), 8),
                    "calls": int(row["calls"] or 0),
                    "failures": int(row["failures"] or 0),
                }
                for row in rows
            ]

    def usage_breakdowns(self, filters: AnalyticsFilters, metric: str) -> Dict[str, List[Dict[str, Any]]]:
        dimensions = {
            "models": ("deployment_id", "deployment_name"),
            "knowledge_bases": ("knowledge_base_id", "knowledge_base_name"),
            "configurations": ("configuration_id", "configuration_label"),
        }
        return {
            name: self._usage_breakdown(filters, key_column, label_column, metric)
            for name, (key_column, label_column) in dimensions.items()
        }

    def usage_events(self, filters: AnalyticsFilters) -> Dict[str, Any]:
        from sqlalchemy import text

        where, params = self._usage_where(filters)
        offset = (filters.page - 1) * filters.page_size
        params = {**params, "limit": filters.page_size, "offset": offset}
        with self.engine.begin() as connection:
            total = int(
                connection.execute(
                    text(f"WITH enriched AS ({self._usage_cte()}) SELECT COUNT(*) FROM enriched {where}"),
                    params,
                ).scalar_one()
            )
            rows = connection.execute(
                text(
                    f"""
                    WITH enriched AS ({self._usage_cte()})
                    SELECT *
                    FROM enriched
                    {where}
                    ORDER BY created_at DESC, id DESC
                    LIMIT :limit OFFSET :offset
                    """
                ),
                params,
            ).mappings()
            items = [self._usage_row(row) for row in rows]
        return _page(items, total, filters)

    def feedback(self, filters: AnalyticsFilters) -> Dict[str, Any]:
        from sqlalchemy import text

        where, params = self._feedback_where(filters)
        params.update({"limit": filters.page_size, "offset": (filters.page - 1) * filters.page_size})
        with self.engine.begin() as connection:
            total = int(
                connection.execute(text(f"SELECT COUNT(*) FROM chat_feedback feedback {where}"), params).scalar_one()
            )
            rows = connection.execute(
                text(
                    f"""
                    SELECT feedback.*
                    FROM chat_feedback feedback
                    {where}
                    ORDER BY feedback.updated_at DESC, feedback.id DESC
                    LIMIT :limit OFFSET :offset
                    """
                ),
                params,
            ).mappings()
            items = [dict(row) for row in rows]
        return _page(items, total, filters)

    def filter_options(self, filters: AnalyticsFilters) -> Dict[str, Any]:
        from sqlalchemy import text

        params = {"from_at": filters.from_at, "to_at": filters.to_at}
        with self.engine.begin() as connection:
            def usage_options(key_column: str, label_column: str) -> List[Dict[str, str]]:
                rows = connection.execute(
                    text(
                        f"""
                        WITH enriched AS ({self._usage_cte()})
                        SELECT DISTINCT {key_column} AS id, {label_column} AS label
                        FROM enriched
                        WHERE created_at >= :from_at AND created_at < :to_at
                          AND COALESCE({key_column}, '') <> ''
                        ORDER BY label, id
                        """
                    ),
                    params,
                ).mappings()
                return [{"id": str(row["id"]), "label": str(row["label"] or row["id"])} for row in rows]

            def knowledge_base_options() -> List[Dict[str, str]]:
                rows = connection.execute(
                    text(
                        f"""
                        WITH enriched AS ({self._usage_cte()})
                        SELECT DISTINCT knowledge_base_id AS id, knowledge_base_name AS label
                        FROM enriched
                        WHERE created_at >= :from_at AND created_at < :to_at
                          AND COALESCE(knowledge_base_id, '') <> ''
                          AND knowledge_base_name <> :unavailable_label
                        ORDER BY label, id
                        """
                    ),
                    {**params, "unavailable_label": UNAVAILABLE_KNOWLEDGE_BASE_LABEL},
                ).mappings()
                options = [{"id": str(row["id"]), "label": str(row["label"] or row["id"])} for row in rows]
                unavailable_count = int(
                    connection.execute(
                        text(
                            f"""
                            WITH enriched AS ({self._usage_cte()})
                            SELECT COUNT(DISTINCT knowledge_base_id)
                            FROM enriched
                            WHERE created_at >= :from_at AND created_at < :to_at
                              AND COALESCE(knowledge_base_id, '') <> ''
                              AND knowledge_base_name = :unavailable_label
                            """
                        ),
                        {**params, "unavailable_label": UNAVAILABLE_KNOWLEDGE_BASE_LABEL},
                    ).scalar_one()
                )
                if unavailable_count:
                    options.append(
                        {
                            "id": UNAVAILABLE_KNOWLEDGE_BASE_ID,
                            "label": (
                                f"{UNAVAILABLE_KNOWLEDGE_BASE_LABEL} "
                                f"({unavailable_count:,} Knowledge Base{'' if unavailable_count == 1 else 's'})"
                            ),
                        }
                    )
                return options

            dimensions = connection.execute(
                text(
                    f"""
                    WITH enriched AS ({self._usage_cte()})
                    SELECT DISTINCT purpose, status, activity_scope
                    FROM enriched
                    WHERE created_at >= :from_at AND created_at < :to_at
                    """
                ),
                params,
            ).mappings()
            dimension_rows = list(dimensions)
            users = list(
                connection.execute(
                    text(
                        """
                        SELECT DISTINCT users.id, users.email, users.first_name, users.last_name
                        FROM users
                        LEFT JOIN chat_messages message ON message.user_id = users.id
                        LEFT JOIN chat_feedback feedback ON feedback.user_id = users.id
                        WHERE users.active = TRUE
                          AND (
                            (message.created_at >= :from_at AND message.created_at < :to_at)
                            OR (feedback.updated_at >= :from_at AND feedback.updated_at < :to_at)
                          )
                        ORDER BY users.email
                        """
                    ),
                    params,
                ).mappings()
            )
            models = usage_options("deployment_id", "deployment_name")
            knowledge_bases = knowledge_base_options()
            configurations = usage_options("configuration_id", "configuration_label")
        return {
            "models": models,
            "knowledge_bases": knowledge_bases,
            "configurations": configurations,
            "purposes": sorted({str(row["purpose"]) for row in dimension_rows if row["purpose"]}),
            "statuses": sorted({str(row["status"]) for row in dimension_rows if row["status"]}),
            "scopes": sorted({str(row["activity_scope"]) for row in dimension_rows if row["activity_scope"]}),
            "users": [
                {
                    "id": row["id"],
                    "label": _user_label(row.get("first_name"), row.get("last_name"), row.get("email")),
                }
                for row in users
            ],
        }

    def upsert_feedback(
        self,
        *,
        assistant_message_id: str,
        version_number: int,
        user_id: str,
        user_name: str,
        user_email: str,
        rating: str,
        comment: Optional[str],
    ) -> Dict[str, Any]:
        from sqlalchemy import text

        now = utc_now()
        with self.engine.begin() as connection:
            assistant = connection.execute(
                text(
                    """
                    SELECT message.*, conversation.knowledge_base_id, conversation.chat_configuration_id
                    FROM chat_messages message
                    JOIN chat_conversations conversation ON conversation.id = message.conversation_id
                    WHERE message.id = :message_id AND message.role = 'assistant'
                    """
                ),
                {"message_id": assistant_message_id},
            ).mappings().first()
            if not assistant:
                raise KeyError(f"Assistant message not found: {assistant_message_id}")
            version = connection.execute(
                text(
                    """
                    SELECT * FROM chat_message_versions
                    WHERE message_id = :message_id AND version_number = :version_number
                    """
                ),
                {"message_id": assistant_message_id, "version_number": version_number},
            ).mappings().first()
            if not version:
                raise KeyError(f"Answer version {version_number} was not found for message {assistant_message_id}.")
            user_message = connection.execute(
                text(
                    """
                    SELECT id, content
                    FROM chat_messages
                    WHERE conversation_id = :conversation_id
                      AND role = 'user'
                      AND created_at <= :assistant_created_at
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """
                ),
                {
                    "conversation_id": assistant["conversation_id"],
                    "assistant_created_at": assistant["created_at"],
                },
            ).mappings().first()
            kb = connection.execute(
                text("SELECT name FROM knowledge_bases WHERE id = :id"),
                {"id": assistant.get("knowledge_base_id") or ""},
            ).mappings().first()
            configuration = connection.execute(
                text("SELECT name, metadata_json FROM chat_configurations WHERE id = :id"),
                {"id": assistant.get("chat_configuration_id") or ""},
            ).mappings().first()
            version_metadata = dict(version.get("metadata_json") or {})
            trace_id = str(version_metadata.get("trace_id") or (assistant.get("metadata_json") or {}).get("trace_id") or "")
            feedback_id = f"feedback-{uuid.uuid4().hex}"
            configuration_metadata = dict(configuration.get("metadata_json") or {}) if configuration else {}
            comment_supplied = comment is not None
            payload = {
                "id": feedback_id,
                "user_id": user_id,
                "conversation_id": assistant["conversation_id"],
                "user_message_id": user_message["id"] if user_message else "",
                "assistant_message_id": assistant_message_id,
                "message_version_id": version["id"],
                "version_number": version_number,
                "knowledge_base_id": assistant.get("knowledge_base_id") or "",
                "chat_configuration_id": assistant.get("chat_configuration_id") or "",
                "trace_id": trace_id,
                "rating": rating,
                "comment": comment or "",
                "comment_supplied": comment_supplied,
                "question_snapshot": user_message["content"] if user_message else str(version_metadata.get("question") or ""),
                "answer_snapshot": version["content"],
                "knowledge_base_name_snapshot": kb["name"] if kb else "",
                "configuration_name_snapshot": configuration["name"] if configuration else "",
                "configuration_code_snapshot": str(configuration_metadata.get("configuration_id") or ""),
                "user_name_snapshot": user_name,
                "user_email_snapshot": user_email,
                "created_at": now,
                "updated_at": now,
            }
            row = connection.execute(
                text(
                    """
                    INSERT INTO chat_feedback (
                        id, user_id, conversation_id, user_message_id, assistant_message_id,
                        message_version_id, version_number, knowledge_base_id, chat_configuration_id,
                        trace_id, rating, comment, question_snapshot, answer_snapshot,
                        knowledge_base_name_snapshot, configuration_name_snapshot,
                        configuration_code_snapshot, user_name_snapshot, user_email_snapshot,
                        created_at, updated_at
                    ) VALUES (
                        :id, :user_id, :conversation_id, :user_message_id, :assistant_message_id,
                        :message_version_id, :version_number, :knowledge_base_id, :chat_configuration_id,
                        :trace_id, :rating, :comment, :question_snapshot, :answer_snapshot,
                        :knowledge_base_name_snapshot, :configuration_name_snapshot,
                        :configuration_code_snapshot, :user_name_snapshot, :user_email_snapshot,
                        :created_at, :updated_at
                    )
                    ON CONFLICT (user_id, assistant_message_id, version_number) DO UPDATE SET
                        rating = EXCLUDED.rating,
                        comment = CASE
                            WHEN :comment_supplied THEN EXCLUDED.comment
                            ELSE chat_feedback.comment
                        END,
                        updated_at = EXCLUDED.updated_at
                    RETURNING *
                    """
                ),
                payload,
            ).mappings().one()
        return dict(row)

    def list_message_feedback(self, message_ids: Sequence[str], user_id: str) -> List[Dict[str, Any]]:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT *
                    FROM chat_feedback
                    WHERE user_id = :user_id
                      AND assistant_message_id = ANY(:message_ids)
                    ORDER BY assistant_message_id, version_number
                    """
                ),
                {"user_id": user_id, "message_ids": list(message_ids)},
            ).mappings()
            return [dict(row) for row in rows]

    def _usage_breakdown(
        self,
        filters: AnalyticsFilters,
        key_column: str,
        label_column: str,
        metric: str,
    ) -> List[Dict[str, Any]]:
        from sqlalchemy import text

        where, params = self._usage_where(filters)
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    f"""
                    WITH enriched AS ({self._usage_cte()})
                    SELECT
                        CASE
                            WHEN {label_column} = '{UNAVAILABLE_KNOWLEDGE_BASE_LABEL}'
                            THEN '{UNAVAILABLE_KNOWLEDGE_BASE_ID}'
                            ELSE COALESCE(NULLIF({key_column}, ''), 'unknown')
                        END AS item_id,
                        COALESCE(NULLIF({label_column}, ''), 'Unknown') AS label,
                        activity_scope,
                        COALESCE(SUM(total_tokens), 0) AS tokens,
                        COALESCE(SUM(estimated_cost_usd), 0) AS cost_usd,
                        COUNT(*) AS calls
                    FROM enriched
                    {where}
                    GROUP BY item_id, label, activity_scope
                    """
                ),
                params,
            ).mappings()
        return _collapse_breakdown([dict(row) for row in rows], metric)

    def _engagement_overview(self, connection: Any, filters: AnalyticsFilters) -> Dict[str, Any]:
        from sqlalchemy import text

        chat_filters = [
            "message.created_at >= :from_at",
            "message.created_at < :to_at",
        ]
        params: Dict[str, Any] = {"from_at": filters.from_at, "to_at": filters.to_at}
        if filters.knowledge_base_id:
            if filters.knowledge_base_id == UNAVAILABLE_KNOWLEDGE_BASE_ID:
                chat_filters.extend(
                    [
                        "COALESCE(conversation.knowledge_base_id, '') <> ''",
                        "NOT EXISTS (SELECT 1 FROM knowledge_bases kb WHERE kb.id = conversation.knowledge_base_id)",
                    ]
                )
            else:
                chat_filters.append("conversation.knowledge_base_id = :knowledge_base_id")
                params["knowledge_base_id"] = filters.knowledge_base_id
        if filters.chat_configuration_id:
            chat_filters.append("conversation.chat_configuration_id = :chat_configuration_id")
            params["chat_configuration_id"] = filters.chat_configuration_id
        if filters.user_id:
            chat_filters.append("COALESCE(message.user_id, conversation.owner_user_id) = :user_id")
            params["user_id"] = filters.user_id
        chat_where = " AND ".join(chat_filters)
        messages = connection.execute(
            text(
                f"""
                SELECT
                    COUNT(*) AS total_messages,
                    COUNT(*) FILTER (WHERE message.role = 'user') AS user_messages,
                    COUNT(*) FILTER (WHERE message.role = 'assistant') AS assistant_messages,
                    COUNT(*) FILTER (WHERE message.status = 'completed') AS completed_messages,
                    COUNT(*) FILTER (WHERE message.status = 'failed') AS failed_messages,
                    COUNT(*) FILTER (WHERE message.status = 'cancelled') AS cancelled_messages,
                    COUNT(DISTINCT message.conversation_id) FILTER (WHERE message.role = 'user') AS active_chats,
                    COUNT(DISTINCT message.user_id) FILTER (
                        WHERE message.role = 'user' AND COALESCE(message.user_id, '') <> ''
                    ) AS active_users,
                    COUNT(*) FILTER (
                        WHERE message.role = 'user' AND COALESCE(message.user_id, '') = ''
                    ) AS unknown_user_messages
                FROM chat_messages message
                JOIN chat_conversations conversation ON conversation.id = message.conversation_id
                WHERE {chat_where}
                """
            ),
            params,
        ).mappings().one()

        feedback_filters = [
            "feedback.updated_at >= :from_at",
            "feedback.updated_at < :to_at",
        ]
        if filters.knowledge_base_id:
            if filters.knowledge_base_id == UNAVAILABLE_KNOWLEDGE_BASE_ID:
                feedback_filters.append(
                    f"COALESCE(feedback.knowledge_base_name_snapshot, '') IN ('', '{UNAVAILABLE_KNOWLEDGE_BASE_LABEL}')"
                )
            else:
                feedback_filters.append("feedback.knowledge_base_id = :knowledge_base_id")
        if filters.chat_configuration_id:
            feedback_filters.append("feedback.chat_configuration_id = :chat_configuration_id")
        if filters.user_id:
            feedback_filters.append("feedback.user_id = :user_id")
        if filters.rating:
            feedback_filters.append("feedback.rating = :rating")
            params["rating"] = filters.rating
        feedback = connection.execute(
            text(
                f"""
                SELECT
                    COUNT(*) AS total_feedback,
                    COUNT(*) FILTER (WHERE rating = 'up') AS thumbs_up,
                    COUNT(*) FILTER (WHERE rating = 'down') AS thumbs_down
                FROM chat_feedback feedback
                WHERE {' AND '.join(feedback_filters)}
                """
            ),
            params,
        ).mappings().one()
        return {
            **{key: int(value or 0) for key, value in messages.items()},
            **{key: int(value or 0) for key, value in feedback.items()},
        }

    @staticmethod
    def _usage_cte() -> str:
        return """
            SELECT
                usage.*,
                COALESCE(deployment.name, usage.model, 'Unknown model') AS deployment_name,
                CASE
                    WHEN COALESCE(usage.knowledge_base_id, '') = '' THEN 'Not attributed'
                    WHEN knowledge.id IS NULL THEN 'Deleted or unavailable'
                    ELSE knowledge.name
                END AS knowledge_base_name,
                COALESCE(
                    NULLIF(usage.chat_configuration_id, ''),
                    evaluation.chat_configuration_id,
                    conversation.chat_configuration_id,
                    ''
                ) AS configuration_id,
                COALESCE(configuration.name, 'Unknown') AS configuration_name,
                COALESCE(configuration.metadata_json ->> 'configuration_id', '') AS configuration_code,
                CASE
                    WHEN COALESCE(usage.evaluation_run_id, '') <> '' THEN 'evaluation'
                    WHEN COALESCE(usage.conversation_id, '') <> '' THEN 'chat'
                    WHEN LOWER(usage.purpose) LIKE 'knowledge_%'
                      OR LOWER(usage.purpose) LIKE 'embedding%'
                      OR LOWER(usage.purpose) LIKE 'document_%'
                      OR LOWER(usage.purpose) LIKE 'index_%'
                      OR LOWER(usage.purpose) LIKE 'ingestion%'
                      OR LOWER(usage.purpose) LIKE 'reindex%' THEN 'knowledge'
                    ELSE 'system'
                END AS activity_scope,
                CASE
                    WHEN COALESCE(configuration.metadata_json ->> 'configuration_id', '') <> ''
                    THEN (configuration.metadata_json ->> 'configuration_id') || ' | ' || configuration.name
                    WHEN configuration.name IS NOT NULL THEN configuration.name
                    ELSE 'Unknown'
                END AS configuration_label
            FROM model_usage_events usage
            LEFT JOIN model_deployments deployment ON deployment.id = usage.deployment_id
            LEFT JOIN knowledge_bases knowledge ON knowledge.id = usage.knowledge_base_id
            LEFT JOIN evaluation_runs evaluation ON evaluation.id = usage.evaluation_run_id
            LEFT JOIN chat_conversations conversation ON conversation.id = usage.conversation_id
            LEFT JOIN chat_configurations configuration ON configuration.id = COALESCE(
                NULLIF(usage.chat_configuration_id, ''),
                evaluation.chat_configuration_id,
                conversation.chat_configuration_id
            )
        """

    @staticmethod
    def _usage_where(filters: AnalyticsFilters) -> Tuple[str, Dict[str, Any]]:
        clauses = ["created_at >= :from_at", "created_at < :to_at"]
        params: Dict[str, Any] = {"from_at": filters.from_at, "to_at": filters.to_at}
        fields = {
            "scope": ("activity_scope", filters.scope),
            "deployment_id": ("deployment_id", filters.deployment_id),
            "chat_configuration_id": ("configuration_id", filters.chat_configuration_id),
            "purpose": ("purpose", filters.purpose),
            "status": ("status", filters.status),
            "user_id": ("user_id", filters.user_id),
        }
        for parameter, (column, value) in fields.items():
            if value:
                clauses.append(f"{column} = :{parameter}")
                params[parameter] = value
        if filters.knowledge_base_id:
            if filters.knowledge_base_id == UNAVAILABLE_KNOWLEDGE_BASE_ID:
                clauses.append("knowledge_base_name = :unavailable_knowledge_base")
                params["unavailable_knowledge_base"] = UNAVAILABLE_KNOWLEDGE_BASE_LABEL
            else:
                clauses.append("knowledge_base_id = :knowledge_base_id")
                params["knowledge_base_id"] = filters.knowledge_base_id
        if filters.query:
            clauses.append(
                """
                LOWER(CONCAT_WS(' ', deployment_name, model, provider, purpose, knowledge_base_name,
                    configuration_label, status, error)) LIKE :query
                """
            )
            params["query"] = f"%{filters.query.lower()}%"
        return f"WHERE {' AND '.join(clauses)}", params

    @staticmethod
    def _feedback_where(filters: AnalyticsFilters) -> Tuple[str, Dict[str, Any]]:
        clauses = ["feedback.updated_at >= :from_at", "feedback.updated_at < :to_at"]
        params: Dict[str, Any] = {"from_at": filters.from_at, "to_at": filters.to_at}
        fields = {
            "chat_configuration_id": filters.chat_configuration_id,
            "user_id": filters.user_id,
            "rating": filters.rating,
        }
        for field, value in fields.items():
            if value:
                clauses.append(f"feedback.{field} = :{field}")
                params[field] = value
        if filters.knowledge_base_id:
            if filters.knowledge_base_id == UNAVAILABLE_KNOWLEDGE_BASE_ID:
                clauses.append(
                    f"COALESCE(feedback.knowledge_base_name_snapshot, '') IN ('', '{UNAVAILABLE_KNOWLEDGE_BASE_LABEL}')"
                )
            else:
                clauses.append("feedback.knowledge_base_id = :knowledge_base_id")
                params["knowledge_base_id"] = filters.knowledge_base_id
        if filters.query:
            clauses.append(
                """
                LOWER(CONCAT_WS(' ', feedback.knowledge_base_name_snapshot,
                    feedback.configuration_name_snapshot, feedback.configuration_code_snapshot,
                    feedback.user_name_snapshot, feedback.user_email_snapshot,
                    feedback.question_snapshot, feedback.answer_snapshot, feedback.comment)) LIKE :query
                """
            )
            params["query"] = f"%{filters.query.lower()}%"
        return f"WHERE {' AND '.join(clauses)}", params

    @staticmethod
    def _usage_row(row: Any) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "deployment_id": row["deployment_id"],
            "deployment_name": row["deployment_name"],
            "provider": row["provider"],
            "model": row["model"],
            "capability": row["capability"],
            "purpose": row["purpose"],
            "activity_scope": row["activity_scope"],
            "status": row["status"],
            "user_id": row.get("user_id") or "",
            "conversation_id": row.get("conversation_id") or "",
            "knowledge_base_id": row.get("knowledge_base_id") or "",
            "knowledge_base_name": row.get("knowledge_base_name") or "Unknown",
            "evaluation_run_id": row.get("evaluation_run_id") or "",
            "chat_configuration_id": row.get("configuration_id") or "",
            "configuration_label": row.get("configuration_label") or "Unknown",
            "input_tokens": int(row.get("input_tokens") or 0),
            "output_tokens": int(row.get("output_tokens") or 0),
            "total_tokens": int(row.get("total_tokens") or 0),
            "latency_ms": float(row.get("latency_ms") or 0),
            "estimated_cost_usd": float(row.get("estimated_cost_usd") or 0),
            "error_code": row.get("error_code") or "",
            "error": row.get("error") or "",
            "created_at": row.get("created_at") or "",
        }


class JsonAnalyticsRepository:
    """Bounded offline analytics over the existing JSON repositories."""

    def __init__(
        self,
        *,
        model_farm_path: str,
        chat_path: str,
        knowledge_path: str,
        evaluation_path: str,
        auth_path: str,
        feedback_path: str,
    ):
        self.model_farm_path = Path(model_farm_path)
        self.chat_path = Path(chat_path)
        self.knowledge_path = Path(knowledge_path)
        self.evaluation_path = Path(evaluation_path)
        self.auth_path = Path(auth_path)
        self.feedback_path = Path(feedback_path)

    def initialize(self) -> None:
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.feedback_path.exists():
            self.feedback_path.write_text(json.dumps({"feedback": {}}, indent=2), encoding="utf-8")

    def overview(self, filters: AnalyticsFilters) -> Dict[str, Any]:
        usage = self._filtered_usage(filters)
        messages, conversations = self._chat_records()
        knowledge = self._read(self.knowledge_path, {"knowledge_bases": {}})
        active_knowledge_base_ids = set(knowledge.get("knowledge_bases", {}))
        feedback = self._filtered_feedback(filters)
        dated_messages = [
            item for item in messages
            if _in_range(item.get("created_at", ""), filters.from_at, filters.to_at)
            and _chat_item_matches(item, conversations, filters, active_knowledge_base_ids)
        ]
        completed = [item for item in usage if item.get("status") == "completed"]
        user_messages = [item for item in dated_messages if item.get("role") == "user"]
        return {
            "usage": {
                "calls": len(usage),
                "completed_calls": len(completed),
                "failed_calls": len(usage) - len(completed),
                "input_tokens": sum(int(item.get("input_tokens") or 0) for item in usage),
                "output_tokens": sum(int(item.get("output_tokens") or 0) for item in usage),
                "total_tokens": sum(int(item.get("total_tokens") or 0) for item in usage),
                "estimated_cost_usd": round(sum(float(item.get("estimated_cost_usd") or 0) for item in usage), 8),
                "average_latency_ms": round(
                    sum(float(item.get("latency_ms") or 0) for item in usage) / len(usage), 3
                ) if usage else 0.0,
            },
            "engagement": {
                "active_chats": len({item.get("conversation_id") for item in user_messages}),
                "total_messages": len(dated_messages),
                "user_messages": len(user_messages),
                "assistant_messages": len([item for item in dated_messages if item.get("role") == "assistant"]),
                "completed_messages": len([item for item in dated_messages if item.get("status") == "completed"]),
                "failed_messages": len([item for item in dated_messages if item.get("status") == "failed"]),
                "cancelled_messages": len([item for item in dated_messages if item.get("status") == "cancelled"]),
                "active_users": len({item.get("user_id") for item in user_messages if item.get("user_id")}),
                "unknown_user_messages": len([item for item in user_messages if not item.get("user_id")]),
                "total_feedback": len(feedback),
                "thumbs_up": len([item for item in feedback if item.get("rating") == "up"]),
                "thumbs_down": len([item for item in feedback if item.get("rating") == "down"]),
            },
            "from": filters.from_at,
            "to": filters.to_at,
        }

    def usage_trend(self, filters: AnalyticsFilters) -> List[Dict[str, Any]]:
        buckets: Dict[str, Dict[str, Any]] = {}
        for item in self._filtered_usage(filters):
            day = str(item.get("created_at") or "")[:10]
            bucket = buckets.setdefault(
                day,
                {"day": day, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0, "calls": 0, "failures": 0},
            )
            bucket["input_tokens"] += int(item.get("input_tokens") or 0)
            bucket["output_tokens"] += int(item.get("output_tokens") or 0)
            bucket["total_tokens"] += int(item.get("total_tokens") or 0)
            bucket["cost_usd"] += float(item.get("estimated_cost_usd") or 0)
            bucket["calls"] += 1
            bucket["failures"] += int(item.get("status") != "completed")
        return [buckets[key] for key in sorted(buckets)]

    def usage_breakdowns(self, filters: AnalyticsFilters, metric: str) -> Dict[str, List[Dict[str, Any]]]:
        usage = self._filtered_usage(filters)
        dimensions = {
            "models": ("deployment_id", "deployment_name"),
            "knowledge_bases": ("knowledge_base_id", "knowledge_base_name"),
            "configurations": ("chat_configuration_id", "configuration_label"),
        }
        result: Dict[str, List[Dict[str, Any]]] = {}
        for name, (key_name, label_name) in dimensions.items():
            rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
            for item in usage:
                key = str(item.get(key_name) or "unknown")
                if name == "knowledge_bases" and item.get(label_name) == UNAVAILABLE_KNOWLEDGE_BASE_LABEL:
                    key = UNAVAILABLE_KNOWLEDGE_BASE_ID
                scope = str(item.get("activity_scope") or "system")
                row = rows.setdefault(
                    (key, scope),
                    {
                        "item_id": key,
                        "label": str(item.get(label_name) or "Unknown"),
                        "activity_scope": scope,
                        "tokens": 0,
                        "cost_usd": 0.0,
                        "calls": 0,
                    },
                )
                row["tokens"] += int(item.get("total_tokens") or 0)
                row["cost_usd"] += float(item.get("estimated_cost_usd") or 0)
                row["calls"] += 1
            result[name] = _collapse_breakdown(list(rows.values()), metric)
        return result

    def usage_events(self, filters: AnalyticsFilters) -> Dict[str, Any]:
        items = sorted(self._filtered_usage(filters), key=lambda item: item.get("created_at", ""), reverse=True)
        return _slice_page(items, filters)

    def feedback(self, filters: AnalyticsFilters) -> Dict[str, Any]:
        items = sorted(self._filtered_feedback(filters), key=lambda item: item.get("updated_at", ""), reverse=True)
        return _slice_page(items, filters)

    def filter_options(self, filters: AnalyticsFilters) -> Dict[str, Any]:
        usage = [
            item for item in self._enriched_usage()
            if _in_range(item.get("created_at", ""), filters.from_at, filters.to_at)
        ]
        auth = self._read(self.auth_path, {"users": {}})
        return {
            "models": _unique_options(usage, "deployment_id", "deployment_name"),
            "knowledge_bases": _knowledge_base_options(usage),
            "configurations": _unique_options(usage, "chat_configuration_id", "configuration_label"),
            "purposes": sorted({str(item.get("purpose")) for item in usage if item.get("purpose")}),
            "statuses": sorted({str(item.get("status")) for item in usage if item.get("status")}),
            "scopes": sorted({str(item.get("activity_scope")) for item in usage if item.get("activity_scope")}),
            "users": [
                {
                    "id": item.get("id"),
                    "label": _user_label(item.get("first_name"), item.get("last_name"), item.get("email")),
                }
                for item in auth.get("users", {}).values()
                if item.get("active", True)
            ],
        }

    def upsert_feedback(
        self,
        *,
        assistant_message_id: str,
        version_number: int,
        user_id: str,
        user_name: str,
        user_email: str,
        rating: str,
        comment: Optional[str],
    ) -> Dict[str, Any]:
        chat = self._read(self.chat_path, {})
        messages = chat.get("chat_messages", {})
        versions = chat.get("chat_message_versions", {})
        assistant = messages.get(assistant_message_id)
        if not assistant or assistant.get("role") != "assistant":
            raise KeyError(f"Assistant message not found: {assistant_message_id}")
        matching_versions = [
            value for value in versions.values()
            if value.get("message_id") == assistant_message_id
            and int(value.get("version_number") or 0) == version_number
        ]
        if not matching_versions:
            raise KeyError(f"Answer version {version_number} was not found for message {assistant_message_id}.")
        version = matching_versions[0]
        conversation = chat.get("chat_conversations", {}).get(assistant.get("conversation_id"), {})
        preceding = [
            item for item in messages.values()
            if item.get("conversation_id") == assistant.get("conversation_id")
            and item.get("role") == "user"
            and item.get("created_at", "") <= assistant.get("created_at", "")
        ]
        preceding.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        user_message = preceding[0] if preceding else {}
        configurations = chat.get("chat_configurations", {})
        configuration = configurations.get(conversation.get("chat_configuration_id"), {})
        knowledge = self._read(self.knowledge_path, {})
        kb = knowledge.get("knowledge_bases", {}).get(conversation.get("knowledge_base_id"), {})
        metadata = version.get("metadata", version.get("metadata_json", {})) or {}
        state = self._read(self.feedback_path, {"feedback": {}})
        existing = next(
            (
                item for item in state["feedback"].values()
                if item.get("user_id") == user_id
                and item.get("assistant_message_id") == assistant_message_id
                and int(item.get("version_number") or 0) == version_number
            ),
            None,
        )
        now = utc_now()
        record = existing or {
            "id": f"feedback-{uuid.uuid4().hex}",
            "user_id": user_id,
            "conversation_id": assistant.get("conversation_id", ""),
            "user_message_id": user_message.get("id", ""),
            "assistant_message_id": assistant_message_id,
            "message_version_id": version.get("id", ""),
            "version_number": version_number,
            "knowledge_base_id": conversation.get("knowledge_base_id", ""),
            "chat_configuration_id": conversation.get("chat_configuration_id", ""),
            "trace_id": metadata.get("trace_id", ""),
            "question_snapshot": user_message.get("content", metadata.get("question", "")),
            "answer_snapshot": version.get("content", ""),
            "knowledge_base_name_snapshot": kb.get("name", ""),
            "configuration_name_snapshot": configuration.get("name", ""),
            "configuration_code_snapshot": (configuration.get("metadata", {}) or {}).get("configuration_id", ""),
            "user_name_snapshot": user_name,
            "user_email_snapshot": user_email,
            "created_at": now,
        }
        record = {
            **record,
            "rating": rating,
            "comment": comment if comment is not None else record.get("comment", ""),
            "updated_at": now,
        }
        state["feedback"][record["id"]] = record
        self.feedback_path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")
        return record

    def list_message_feedback(self, message_ids: Sequence[str], user_id: str) -> List[Dict[str, Any]]:
        requested = set(message_ids)
        state = self._read(self.feedback_path, {"feedback": {}})
        return [
            dict(item)
            for item in state.get("feedback", {}).values()
            if item.get("user_id") == user_id and item.get("assistant_message_id") in requested
        ]

    def _filtered_usage(self, filters: AnalyticsFilters) -> List[Dict[str, Any]]:
        return [item for item in self._enriched_usage() if _usage_matches(item, filters)]

    def _filtered_feedback(self, filters: AnalyticsFilters) -> List[Dict[str, Any]]:
        state = self._read(self.feedback_path, {"feedback": {}})
        return [item for item in state["feedback"].values() if _feedback_matches(item, filters)]

    def _enriched_usage(self) -> List[Dict[str, Any]]:
        model = self._read(self.model_farm_path, {"usage": [], "deployments": {}})
        chat = self._read(self.chat_path, {})
        knowledge = self._read(self.knowledge_path, {})
        evaluation = self._read(self.evaluation_path, {})
        deployments = model.get("deployments", {})
        conversations = chat.get("chat_conversations", {})
        configurations = chat.get("chat_configurations", {})
        knowledge_bases = knowledge.get("knowledge_bases", {})
        runs = evaluation.get("evaluation_runs", {})
        items = []
        for raw in model.get("usage", []):
            item = dict(raw)
            run = runs.get(item.get("evaluation_run_id"), {})
            conversation = conversations.get(item.get("conversation_id"), {})
            configuration_id = (
                item.get("chat_configuration_id")
                or run.get("chat_configuration_id")
                or conversation.get("chat_configuration_id")
                or ""
            )
            configuration = configurations.get(configuration_id, {})
            code = (configuration.get("metadata", {}) or {}).get("configuration_id", "")
            configuration_name = configuration.get("name", "Unknown")
            item.update(
                {
                    "deployment_name": deployments.get(item.get("deployment_id"), {}).get("name", item.get("model", "Unknown")),
                    "knowledge_base_name": (
                        knowledge_bases.get(item.get("knowledge_base_id"), {}).get("name")
                        or (
                            UNAVAILABLE_KNOWLEDGE_BASE_LABEL
                            if item.get("knowledge_base_id")
                            else "Not attributed"
                        )
                    ),
                    "chat_configuration_id": configuration_id,
                    "configuration_label": f"{code} | {configuration_name}" if code else configuration_name,
                    "activity_scope": _usage_scope(item),
                }
            )
            items.append(item)
        return items

    def _chat_records(self) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        state = self._read(self.chat_path, {})
        return list(state.get("chat_messages", {}).values()), state.get("chat_conversations", {})

    @staticmethod
    def _read(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
        if not path.exists():
            return json.loads(json.dumps(default))
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return json.loads(json.dumps(default))


def default_analytics_range(days: int = 30) -> Tuple[str, str]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=max(1, days))
    return start.isoformat(), end.isoformat()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_utc(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise AnalyticsError(f"Invalid {label} timestamp.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _usage_scope(item: Dict[str, Any]) -> str:
    if item.get("evaluation_run_id"):
        return "evaluation"
    if item.get("conversation_id"):
        return "chat"
    purpose = str(item.get("purpose") or "").lower()
    if purpose.startswith(KNOWLEDGE_PURPOSE_PREFIXES):
        return "knowledge"
    return "system"


def _usage_matches(item: Dict[str, Any], filters: AnalyticsFilters) -> bool:
    if not _in_range(item.get("created_at", ""), filters.from_at, filters.to_at):
        return False
    checks = (
        ("activity_scope", filters.scope),
        ("deployment_id", filters.deployment_id),
        ("chat_configuration_id", filters.chat_configuration_id),
        ("purpose", filters.purpose),
        ("status", filters.status),
        ("user_id", filters.user_id),
    )
    if any(value and str(item.get(key) or "") != value for key, value in checks):
        return False
    if filters.knowledge_base_id:
        if filters.knowledge_base_id == UNAVAILABLE_KNOWLEDGE_BASE_ID:
            if item.get("knowledge_base_name") != UNAVAILABLE_KNOWLEDGE_BASE_LABEL:
                return False
        elif str(item.get("knowledge_base_id") or "") != filters.knowledge_base_id:
            return False
    if filters.query:
        haystack = " ".join(
            str(item.get(key) or "")
            for key in (
                "deployment_name", "model", "provider", "purpose", "knowledge_base_name",
                "configuration_label", "status", "error",
            )
        ).lower()
        if filters.query.lower() not in haystack:
            return False
    return True


def _feedback_matches(item: Dict[str, Any], filters: AnalyticsFilters) -> bool:
    if not _in_range(item.get("updated_at", ""), filters.from_at, filters.to_at):
        return False
    checks = (
        ("chat_configuration_id", filters.chat_configuration_id),
        ("user_id", filters.user_id),
        ("rating", filters.rating),
    )
    if any(value and str(item.get(key) or "") != value for key, value in checks):
        return False
    if filters.knowledge_base_id:
        if filters.knowledge_base_id == UNAVAILABLE_KNOWLEDGE_BASE_ID:
            if str(item.get("knowledge_base_name_snapshot") or "") not in {
                "",
                UNAVAILABLE_KNOWLEDGE_BASE_LABEL,
            }:
                return False
        elif str(item.get("knowledge_base_id") or "") != filters.knowledge_base_id:
            return False
    if filters.query:
        haystack = " ".join(
            str(item.get(key) or "")
            for key in (
                "knowledge_base_name_snapshot", "configuration_name_snapshot",
                "configuration_code_snapshot", "user_name_snapshot", "user_email_snapshot",
                "question_snapshot", "answer_snapshot", "comment",
            )
        ).lower()
        if filters.query.lower() not in haystack:
            return False
    return True


def _chat_item_matches(
    message: Dict[str, Any],
    conversations: Dict[str, Dict[str, Any]],
    filters: AnalyticsFilters,
    active_knowledge_base_ids: Optional[set[str]] = None,
) -> bool:
    conversation = conversations.get(message.get("conversation_id"), {})
    if filters.knowledge_base_id:
        conversation_knowledge_base_id = str(conversation.get("knowledge_base_id") or "")
        if filters.knowledge_base_id == UNAVAILABLE_KNOWLEDGE_BASE_ID:
            if (
                not conversation_knowledge_base_id
                or conversation_knowledge_base_id in (active_knowledge_base_ids or set())
            ):
                return False
        elif conversation_knowledge_base_id != filters.knowledge_base_id:
            return False
    if filters.chat_configuration_id and conversation.get("chat_configuration_id") != filters.chat_configuration_id:
        return False
    if filters.user_id and (message.get("user_id") or conversation.get("owner_user_id")) != filters.user_id:
        return False
    return True


def _in_range(value: str, start: str, end: str) -> bool:
    return bool(value and start <= value < end)


def _collapse_breakdown(rows: Sequence[Dict[str, Any]], metric: str) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("item_id") or "unknown")
        bucket = grouped.setdefault(
            key,
            {
                "id": key,
                "label": str(row.get("label") or "Unknown"),
                "tokens": 0,
                "cost_usd": 0.0,
                "calls": 0,
                "chat": 0.0,
                "evaluation": 0.0,
                "knowledge": 0.0,
                "system": 0.0,
            },
        )
        tokens = int(row.get("tokens") or 0)
        cost = float(row.get("cost_usd") or 0)
        bucket["tokens"] += tokens
        bucket["cost_usd"] += cost
        bucket["calls"] += int(row.get("calls") or 0)
        scope = str(row.get("activity_scope") or "system")
        bucket[scope] += float(tokens if metric == "tokens" else cost)
    ordered = sorted(
        grouped.values(),
        key=lambda item: item["tokens"] if metric == "tokens" else item["cost_usd"],
        reverse=True,
    )
    if len(ordered) <= 10:
        return ordered
    other = {
        "id": "other",
        "label": "Other",
        "tokens": 0,
        "cost_usd": 0.0,
        "calls": 0,
        "chat": 0.0,
        "evaluation": 0.0,
        "knowledge": 0.0,
        "system": 0.0,
    }
    for item in ordered[10:]:
        for key in ("tokens", "cost_usd", "calls", "chat", "evaluation", "knowledge", "system"):
            other[key] += item[key]
    return [*ordered[:10], other]


def _unique_options(rows: Iterable[Dict[str, Any]], id_key: str, label_key: str) -> List[Dict[str, str]]:
    options = {
        str(row.get(id_key) or ""): str(row.get(label_key) or "Unknown")
        for row in rows
        if row.get(id_key)
    }
    return [{"id": key, "label": options[key]} for key in sorted(options, key=lambda item: options[item].lower())]


def _knowledge_base_options(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    available: Dict[str, str] = {}
    unavailable_ids = set()
    for row in rows:
        knowledge_base_id = str(row.get("knowledge_base_id") or "")
        if not knowledge_base_id:
            continue
        label = str(row.get("knowledge_base_name") or UNAVAILABLE_KNOWLEDGE_BASE_LABEL)
        if label == UNAVAILABLE_KNOWLEDGE_BASE_LABEL:
            unavailable_ids.add(knowledge_base_id)
        else:
            available[knowledge_base_id] = label
    options = [
        {"id": key, "label": available[key]}
        for key in sorted(available, key=lambda item: available[item].lower())
    ]
    if unavailable_ids:
        count = len(unavailable_ids)
        options.append(
            {
                "id": UNAVAILABLE_KNOWLEDGE_BASE_ID,
                "label": (
                    f"{UNAVAILABLE_KNOWLEDGE_BASE_LABEL} "
                    f"({count:,} Knowledge Base{'' if count == 1 else 's'})"
                ),
            }
        )
    return options


def _user_label(first_name: Any, last_name: Any, email: Any) -> str:
    name = " ".join(str(value or "").strip() for value in (first_name, last_name)).strip()
    return name or str(email or "Unknown user")


def _page(items: List[Dict[str, Any]], total: int, filters: AnalyticsFilters) -> Dict[str, Any]:
    return {
        "items": items,
        "total": total,
        "page": filters.page,
        "page_size": filters.page_size,
        "pages": max(1, (total + filters.page_size - 1) // filters.page_size),
    }


def _slice_page(items: List[Dict[str, Any]], filters: AnalyticsFilters) -> Dict[str, Any]:
    offset = (filters.page - 1) * filters.page_size
    return _page(items[offset: offset + filters.page_size], len(items), filters)
