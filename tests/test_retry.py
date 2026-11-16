
import pytest

import src.utils as utils


def test_retry_succeeds_on_first_attempt(monkeypatch):
    calls = []

    def operation():
        calls.append(1)
        return "SUCCESS"

    monkeypatch.setattr(utils.time, "sleep", lambda delay: None)

    result = utils.retry_with_backoff(operation)

    assert result == "SUCCESS"
    assert len(calls) == 1


def test_retry_succeeds_after_temporary_failures(monkeypatch):
    calls = []
    delays = []

    def operation():
        calls.append(1)

        if len(calls) < 3:
            raise TimeoutError("Temporary API timeout")

        return "SUCCESS"

    monkeypatch.setattr(utils.time, "sleep", delays.append)

    result = utils.retry_with_backoff(
        operation,
        max_retries=3,
        base_delay=1.0,
        exceptions=(TimeoutError,),
    )

    assert result == "SUCCESS"
    assert len(calls) == 3
    assert delays == [1.0, 2.0]


def test_retry_raises_original_exception_after_all_attempts(monkeypatch):
    calls = []
    delays = []
    original_error = TimeoutError("API unavailable")

    def operation():
        calls.append(1)
        raise original_error

    monkeypatch.setattr(utils.time, "sleep", delays.append)

    with pytest.raises(TimeoutError) as error:
        utils.retry_with_backoff(
            operation,
            max_retries=3,
            base_delay=1.0,
            exceptions=(TimeoutError,),
        )

    assert error.value is original_error
    assert len(calls) == 3
    assert delays == [1.0, 2.0]


def test_retry_does_not_retry_unconfigured_exception(monkeypatch):
    calls = []

    def operation():
        calls.append(1)
        raise ValueError("Invalid input")

    monkeypatch.setattr(
        utils.time,
        "sleep",
        lambda delay: pytest.fail("Unexpected retry"),
    )

    with pytest.raises(ValueError, match="Invalid input"):
        utils.retry_with_backoff(
            operation,
            max_retries=3,
            exceptions=(TimeoutError,),
        )

    assert len(calls) == 1


@pytest.mark.parametrize("max_retries", [0, -1])
def test_retry_rejects_invalid_attempt_count(max_retries):
    with pytest.raises(ValueError, match="max_retries"):
        utils.retry_with_backoff(
            lambda: "SUCCESS",
            max_retries=max_retries,
        )


def test_retry_rejects_negative_base_delay():
    with pytest.raises(ValueError, match="base_delay"):
        utils.retry_with_backoff(
            lambda: "SUCCESS",
            base_delay=-1.0,
        )
