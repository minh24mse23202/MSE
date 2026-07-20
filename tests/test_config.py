from aragbiz.config import load_config


def test_conversation_history_environment_sets_server_maxima(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[chat]\n"
        "history_default_exchanges = 3\n"
        "history_default_characters = 4000\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ARAGBIZ_CONVERSATION_MAX_EXCHANGES", "6")
    monkeypatch.setenv("ARAGBIZ_CONVERSATION_MAX_CHARACTERS", "10000")

    config = load_config(config_path)

    assert config.conversation_history_default_exchanges == 3
    assert config.conversation_history_default_characters == 4000
    assert config.conversation_history_max_exchanges == 6
    assert config.conversation_history_max_characters == 10000


def test_conversation_history_defaults_are_clamped_to_environment_maxima(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[chat]\n"
        "history_default_exchanges = 5\n"
        "history_default_characters = 8000\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ARAGBIZ_CONVERSATION_MAX_EXCHANGES", "2")
    monkeypatch.setenv("ARAGBIZ_CONVERSATION_MAX_CHARACTERS", "2000")

    config = load_config(config_path)

    assert config.conversation_history_default_exchanges == 2
    assert config.conversation_history_default_characters == 2000
