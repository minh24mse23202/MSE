import pytest

from aragbiz.agent import AgentActionError, AgentToolRegistry, parse_agent_action


def test_agent_registry_hides_unavailable_tools_from_planner():
    registry = AgentToolRegistry()

    planner_names = registry.available_names(public_web_enabled=False)
    listed = {tool.name: tool for tool in registry.list_tools(public_web_enabled=False)}

    assert "search_knowledge_base" in planner_names
    assert "finish" in planner_names
    assert "fetch_public_url" not in planner_names
    assert listed["fetch_public_url"].available is False
    assert listed["search_google_drive"].available is False
    assert "search_google_drive" not in planner_names


def test_agent_action_requires_one_available_json_action():
    action = parse_agent_action(
        '{"action":"search_knowledge_base","arguments":{"query":"approval"},"reason":"Need evidence"}',
        {"search_knowledge_base", "finish"},
    )
    assert action.action == "search_knowledge_base"
    assert action.arguments["query"] == "approval"

    with pytest.raises(AgentActionError, match="unavailable"):
        parse_agent_action('{"action":"query_database","arguments":{}}', {"finish"})
