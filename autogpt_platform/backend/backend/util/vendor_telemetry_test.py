import os

from backend.util.vendor_telemetry import VENDOR_TELEMETRY_OFF, disable_vendor_telemetry


def test_turns_vendor_telemetry_off_when_unset():
    environ: dict[str, str] = {}

    disable_vendor_telemetry(environ)

    assert environ == {
        "GRAPHITI_TELEMETRY_ENABLED": "false",
        "MEM0_TELEMETRY": "false",
    }


def test_keeps_a_value_the_deployment_set():
    environ = {"MEM0_TELEMETRY": "true"}

    disable_vendor_telemetry(environ)

    assert environ["MEM0_TELEMETRY"] == "true"
    assert environ["GRAPHITI_TELEMETRY_ENABLED"] == "false"


def test_importing_backend_sets_the_switches():
    for name in VENDOR_TELEMETRY_OFF:
        assert name in os.environ


def test_graphiti_reads_the_variable_we_set():
    from graphiti_core.telemetry.telemetry import TELEMETRY_ENV_VAR

    assert TELEMETRY_ENV_VAR in VENDOR_TELEMETRY_OFF


def test_mem0_telemetry_client_is_disabled():
    from mem0.memory import telemetry

    assert telemetry.MEM0_TELEMETRY is False
    assert telemetry.client_telemetry.posthog.disabled is True
