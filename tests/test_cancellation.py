import pytest

from aragbiz.cancellation import AnswerCancelled, CancellationCoordinator


def test_cancellation_coordinator_tracks_active_and_terminal_requests():
    coordinator = CancellationCoordinator()
    token = coordinator.register("request-1")

    assert coordinator.status("request-1") == "active"
    assert coordinator.request_cancel("request-1", "Stopped in test.") == "cancel_requested"
    assert token.is_cancelled is True
    with pytest.raises(AnswerCancelled, match="Stopped in test"):
        token.raise_if_cancelled()

    coordinator.finish("request-1", "cancelled")
    assert coordinator.status("request-1") == "cancelled"
    with pytest.raises(ValueError, match="already cancelled"):
        coordinator.request_cancel("request-1")


def test_cancellation_coordinator_rejects_duplicate_active_request():
    coordinator = CancellationCoordinator()
    coordinator.register("request-1")

    with pytest.raises(ValueError, match="already active"):
        coordinator.register("request-1")
    with pytest.raises(KeyError):
        coordinator.request_cancel("request-missing")
