import pytest

from headline_arena import build_payload


def test_build_payload_is_dry_run_and_direction_explicit():
    payload = build_payload(target="gold", probability_positive=0.7,
                            forecast_time="2026-09-09T00:00:00+00:00")
    assert payload["direction"] == "up"
    assert payload["mode"] == "dry-run"
    assert payload["network_submission"] is False


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_probability_is_bounded(value):
    with pytest.raises(ValueError):
        build_payload(target="gold", probability_positive=value)
