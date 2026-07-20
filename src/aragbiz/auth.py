from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol


class AuthenticationError(ValueError):
    """Raised when local authentication or authorization fails."""


@dataclass(frozen=True)
class UserRecord:
    id: str
    email: str
    password_hash: str
    first_name: str = ""
    last_name: str = ""
    role: str = "user"
    active: bool = True
    created_at: str = ""
    updated_at: str = ""


class AuthRepository(Protocol):
    def initialize(self) -> None: ...

    def list_users(self) -> List[UserRecord]: ...

    def get_user(self, user_id: str) -> UserRecord: ...

    def get_by_email(self, email: str) -> UserRecord: ...

    def save_user(self, user: UserRecord) -> UserRecord: ...


class JsonAuthRepository:
    def __init__(self, path: str):
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(json.dumps({"users": {}}, indent=2), encoding="utf-8")

    def list_users(self) -> List[UserRecord]:
        return [_user_from_dict(item) for item in self._read()["users"].values()]

    def get_user(self, user_id: str) -> UserRecord:
        payload = self._read()["users"].get(user_id)
        if not payload:
            raise KeyError(f"User not found: {user_id}")
        return _user_from_dict(payload)

    def get_by_email(self, email: str) -> UserRecord:
        normalized = normalize_email(email)
        for user in self.list_users():
            if user.email == normalized:
                return user
        raise KeyError(f"User not found: {normalized}")

    def save_user(self, user: UserRecord) -> UserRecord:
        state = self._read()
        state["users"][user.id] = asdict(user)
        self.path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")
        return user

    def _read(self) -> Dict[str, Any]:
        self.initialize()
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            state = {}
        state.setdefault("users", {})
        return state


class PostgresAuthRepository:
    def __init__(self, database_url: str):
        try:
            from sqlalchemy import create_engine  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise AuthenticationError("Install the api extra to use PostgreSQL authentication.") from exc
        self.engine = create_engine(database_url, future=True)

    def initialize(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            first_name TEXT NOT NULL DEFAULT '',
            last_name TEXT NOT NULL DEFAULT '',
            role TEXT NOT NULL DEFAULT 'user',
            active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
        """
        with self.engine.begin() as connection:
            for statement in [part.strip() for part in ddl.split(";") if part.strip()]:
                connection.exec_driver_sql(statement)

    def list_users(self) -> List[UserRecord]:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            rows = connection.execute(text("SELECT * FROM users ORDER BY created_at")).mappings()
            return [_user_from_row(row) for row in rows]

    def get_user(self, user_id: str) -> UserRecord:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            row = connection.execute(text("SELECT * FROM users WHERE id = :id"), {"id": user_id}).mappings().first()
        if not row:
            raise KeyError(f"User not found: {user_id}")
        return _user_from_row(row)

    def get_by_email(self, email: str) -> UserRecord:
        from sqlalchemy import text

        normalized = normalize_email(email)
        with self.engine.begin() as connection:
            row = connection.execute(text("SELECT * FROM users WHERE email = :email"), {"email": normalized}).mappings().first()
        if not row:
            raise KeyError(f"User not found: {normalized}")
        return _user_from_row(row)

    def save_user(self, user: UserRecord) -> UserRecord:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO users (id, email, password_hash, first_name, last_name, role, active, created_at, updated_at)
                    VALUES (:id, :email, :password_hash, :first_name, :last_name, :role, :active, :created_at, :updated_at)
                    ON CONFLICT (id) DO UPDATE SET
                        email = EXCLUDED.email, password_hash = EXCLUDED.password_hash,
                        first_name = EXCLUDED.first_name, last_name = EXCLUDED.last_name,
                        role = EXCLUDED.role, active = EXCLUDED.active, updated_at = EXCLUDED.updated_at
                    """
                ),
                asdict(user),
            )
        return user


class AuthService:
    def __init__(
        self,
        repository: AuthRepository,
        *,
        jwt_secret: str,
        token_ttl_seconds: int = 8 * 60 * 60,
        auth_required: bool = False,
    ):
        self.repository = repository
        self.jwt_secret = jwt_secret or "aragbiz-local-development-secret"
        self.token_ttl_seconds = max(int(token_ttl_seconds), 300)
        self.auth_required = auth_required
        self.repository.initialize()
        self._bootstrap_admin()

    def signup(self, email: str, password: str, first_name: str = "", last_name: str = "") -> UserRecord:
        normalized = normalize_email(email)
        if not normalized or "@" not in normalized:
            raise AuthenticationError("Enter a valid email address.")
        if len(password) < 8:
            raise AuthenticationError("Password must contain at least 8 characters.")
        try:
            self.repository.get_by_email(normalized)
        except KeyError:
            pass
        else:
            raise AuthenticationError("An account with this email already exists.")
        now = utc_now()
        role = "admin" if not self.repository.list_users() else "user"
        return self.repository.save_user(
            UserRecord(
                id=f"user-{uuid.uuid4().hex}",
                email=normalized,
                password_hash=hash_password(password),
                first_name=first_name.strip(),
                last_name=last_name.strip(),
                role=role,
                active=True,
                created_at=now,
                updated_at=now,
            )
        )

    def login(self, email: str, password: str) -> tuple[UserRecord, str]:
        try:
            user = self.repository.get_by_email(email)
        except KeyError as exc:
            raise AuthenticationError("Invalid email or password.") from exc
        if not user.active or not verify_password(password, user.password_hash):
            raise AuthenticationError("Invalid email or password.")
        return user, self.issue_token(user)

    def update_profile(
        self,
        user_id: str,
        *,
        email: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        current_password: str = "",
        new_password: Optional[str] = None,
    ) -> UserRecord:
        try:
            user = self.repository.get_user(user_id)
        except KeyError as exc:
            raise AuthenticationError("The user account no longer exists.") from exc
        if not user.active:
            raise AuthenticationError("This account is disabled.")

        updated_email = user.email if email is None else normalize_email(email)
        if not updated_email or "@" not in updated_email:
            raise AuthenticationError("Enter a valid email address.")
        email_changed = updated_email != user.email
        password_changed = new_password is not None
        if email_changed or password_changed:
            if not current_password or not verify_password(current_password, user.password_hash):
                raise AuthenticationError("Current password is incorrect.")
        if password_changed and len(new_password or "") < 8:
            raise AuthenticationError("New password must contain at least 8 characters.")
        if email_changed:
            try:
                existing = self.repository.get_by_email(updated_email)
            except KeyError:
                pass
            else:
                if existing.id != user.id:
                    raise AuthenticationError("An account with this email already exists.")

        return self.repository.save_user(
            replace(
                user,
                email=updated_email,
                password_hash=hash_password(new_password) if password_changed else user.password_hash,
                first_name=user.first_name if first_name is None else first_name.strip(),
                last_name=user.last_name if last_name is None else last_name.strip(),
                updated_at=utc_now(),
            )
        )

    def issue_token(self, user: UserRecord) -> str:
        now = int(time.time())
        return encode_token(
            {
                "sub": user.id,
                "email": user.email,
                "role": user.role,
                "iat": now,
                "exp": now + self.token_ttl_seconds,
            },
            self.jwt_secret,
        )

    def current_user(self, authorization: str = "") -> UserRecord:
        if not authorization:
            if self.auth_required:
                raise AuthenticationError("Authentication is required.")
            return UserRecord("dev-admin", "local@aragbiz.dev", "", "Local", "Admin", "admin", True)
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise AuthenticationError("Use a Bearer access token.")
        payload = decode_token(token, self.jwt_secret)
        try:
            user = self.repository.get_user(str(payload.get("sub") or ""))
        except KeyError as exc:
            raise AuthenticationError("The access token user no longer exists.") from exc
        if not user.active:
            raise AuthenticationError("This account is disabled.")
        return user

    def require_admin(self, authorization: str = "") -> UserRecord:
        user = self.current_user(authorization)
        if user.role != "admin":
            raise AuthenticationError("Administrator role is required.")
        return user

    def _bootstrap_admin(self) -> None:
        email = os.getenv("ARAGBIZ_BOOTSTRAP_ADMIN_EMAIL", "").strip()
        password = os.getenv("ARAGBIZ_BOOTSTRAP_ADMIN_PASSWORD", "")
        if not email or not password or self.repository.list_users():
            return
        user = self.signup(email, password, "System", "Administrator")
        if user.role != "admin":
            raise AuthenticationError("Unable to bootstrap the administrator account.")


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    rounds = 310_000
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    return f"pbkdf2_sha256${rounds}${_b64encode(salt)}${_b64encode(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, rounds, salt, expected = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), _b64decode(salt), int(rounds))
        return hmac.compare_digest(actual, _b64decode(expected))
    except (ValueError, TypeError):
        return False


def encode_token(payload: Dict[str, Any], secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    segments = [_json_segment(header), _json_segment(payload)]
    signing_input = ".".join(segments).encode("ascii")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return ".".join([*segments, _b64encode(signature)])


def decode_token(token: str, secret: str) -> Dict[str, Any]:
    try:
        header_segment, payload_segment, signature_segment = token.split(".", 2)
        signing_input = f"{header_segment}.{payload_segment}".encode("ascii")
        expected = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _b64decode(signature_segment)):
            raise AuthenticationError("Invalid access token signature.")
        payload = json.loads(_b64decode(payload_segment).decode("utf-8"))
        if int(payload.get("exp") or 0) <= int(time.time()):
            raise AuthenticationError("The access token has expired.")
        return payload
    except AuthenticationError:
        raise
    except Exception as exc:
        raise AuthenticationError("Invalid access token.") from exc


def normalize_email(email: str) -> str:
    return email.strip().lower()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_segment(value: Dict[str, Any]) -> str:
    return _b64encode(json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8"))


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _user_from_dict(payload: Dict[str, Any]) -> UserRecord:
    return UserRecord(**payload)


def _user_from_row(row: Any) -> UserRecord:
    return UserRecord(
        id=row["id"], email=row["email"], password_hash=row["password_hash"], first_name=row.get("first_name") or "",
        last_name=row.get("last_name") or "", role=row.get("role") or "user", active=bool(row.get("active", True)),
        created_at=row.get("created_at") or "", updated_at=row.get("updated_at") or "",
    )
