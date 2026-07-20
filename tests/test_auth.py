import pytest

from aragbiz.auth import AuthenticationError, AuthService, JsonAuthRepository


def build_auth_service(tmp_path):
    return AuthService(
        JsonAuthRepository(str(tmp_path / "auth.json")),
        jwt_secret="test-jwt-secret",
        auth_required=True,
    )


def test_profile_names_can_be_updated_without_password(tmp_path, monkeypatch):
    monkeypatch.delenv("ARAGBIZ_BOOTSTRAP_ADMIN_EMAIL", raising=False)
    monkeypatch.delenv("ARAGBIZ_BOOTSTRAP_ADMIN_PASSWORD", raising=False)
    service = build_auth_service(tmp_path)
    user = service.signup("user@example.com", "old-password", "Old", "Name")

    updated = service.update_profile(user.id, first_name="New", last_name="Person")

    assert updated.first_name == "New"
    assert updated.last_name == "Person"
    assert updated.email == user.email
    assert updated.role == "admin"


def test_profile_email_change_requires_current_password(tmp_path, monkeypatch):
    monkeypatch.delenv("ARAGBIZ_BOOTSTRAP_ADMIN_EMAIL", raising=False)
    monkeypatch.delenv("ARAGBIZ_BOOTSTRAP_ADMIN_PASSWORD", raising=False)
    service = build_auth_service(tmp_path)
    user = service.signup("old@example.com", "old-password")

    with pytest.raises(AuthenticationError, match="Current password"):
        service.update_profile(user.id, email="new@example.com")

    updated = service.update_profile(
        user.id,
        email="new@example.com",
        current_password="old-password",
    )
    assert updated.email == "new@example.com"


def test_profile_rejects_duplicate_email(tmp_path, monkeypatch):
    monkeypatch.delenv("ARAGBIZ_BOOTSTRAP_ADMIN_EMAIL", raising=False)
    monkeypatch.delenv("ARAGBIZ_BOOTSTRAP_ADMIN_PASSWORD", raising=False)
    service = build_auth_service(tmp_path)
    user = service.signup("first@example.com", "first-password")
    service.signup("second@example.com", "second-password")

    with pytest.raises(AuthenticationError, match="already exists"):
        service.update_profile(
            user.id,
            email="second@example.com",
            current_password="first-password",
        )


def test_profile_password_change_replaces_login_password(tmp_path, monkeypatch):
    monkeypatch.delenv("ARAGBIZ_BOOTSTRAP_ADMIN_EMAIL", raising=False)
    monkeypatch.delenv("ARAGBIZ_BOOTSTRAP_ADMIN_PASSWORD", raising=False)
    service = build_auth_service(tmp_path)
    user = service.signup("user@example.com", "old-password")

    service.update_profile(
        user.id,
        current_password="old-password",
        new_password="new-password",
    )

    with pytest.raises(AuthenticationError, match="Invalid email or password"):
        service.login("user@example.com", "old-password")
    logged_in, token = service.login("user@example.com", "new-password")
    assert logged_in.id == user.id
    assert token


def test_profile_api_returns_refreshed_token(tmp_path, monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import api.main as api_main

    monkeypatch.delenv("ARAGBIZ_BOOTSTRAP_ADMIN_EMAIL", raising=False)
    monkeypatch.delenv("ARAGBIZ_BOOTSTRAP_ADMIN_PASSWORD", raising=False)
    service = build_auth_service(tmp_path)
    user = service.signup("profile@example.com", "old-password", "Profile", "User")
    monkeypatch.setattr(api_main, "auth_service", service)
    client = TestClient(api_main.app)
    headers = {"Authorization": f"Bearer {service.issue_token(user)}"}

    response = client.patch(
        "/auth/me",
        headers=headers,
        json={
            "first_name": "Updated",
            "last_name": "Account",
            "email": "updated@example.com",
            "current_password": "old-password",
            "new_password": "new-password",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["access_token"]
    assert payload["user"]["email"] == "updated@example.com"
    assert payload["user"]["first_name"] == "Updated"
    refreshed_headers = {"Authorization": f"Bearer {payload['access_token']}"}
    assert client.get("/auth/me", headers=refreshed_headers).json()["last_name"] == "Account"
    assert client.post(
        "/auth/login",
        json={"email": "updated@example.com", "password": "new-password"},
    ).status_code == 200
