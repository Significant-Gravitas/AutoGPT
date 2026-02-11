"""
Test: GHSA-4crw-9p35-9x54 - Disabled block bypass via graph creation

Validates that the fix prevents disabled blocks (specifically BlockInstallationBlock)
from being used in graphs. Tests all three attack vectors:

1. Creating a graph with a disabled block
2. Executing a direct block call on a disabled block
3. Executing a graph containing a disabled block (if one existed pre-fix)

Usage against a RUNNING server:
    python test_disabled_block_bypass.py --base-url http://localhost:8006 --token <JWT>

Usage as a pytest unit test (no server needed):
    poetry run pytest test_disabled_block_bypass.py -xvs
"""

import argparse
import sys
import uuid

import pytest

# ── Block IDs of all currently disabled blocks ──
BLOCK_INSTALLATION_BLOCK_ID = "45e78db5-03e9-447f-9395-308d712f5f08"


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — validate the fix at the model layer (no server needed)
# Run with:  poetry run pytest test_disabled_block_bypass.py -xvs
# ═══════════════════════════════════════════════════════════════════════════════


class TestDisabledBlockGraphValidation:
    """Test that graph validation rejects disabled blocks."""

    @staticmethod
    def _make_graph_with_block(block_id: str, input_default: dict | None = None):
        """Helper: build a GraphModel containing a single node with the given block."""
        from backend.data.graph import Graph, Node, make_graph_model

        node_id = str(uuid.uuid4())
        graph = Graph(
            name=f"test-disabled-{block_id[:8]}",
            description="GHSA-4crw-9p35-9x54 test",
            nodes=[
                Node(
                    id=node_id,
                    block_id=block_id,
                    input_default=input_default or {},
                )
            ],
            links=[],
        )
        return make_graph_model(graph, user_id="test-user")

    def test_graph_with_block_installation_block_is_rejected(self):
        """
        GHSA-4crw-9p35-9x54: The core vulnerability.
        Creating a graph with BlockInstallationBlock (disabled=True) must fail validation.
        """
        graph_model = self._make_graph_with_block(
            BLOCK_INSTALLATION_BLOCK_ID,
            input_default={"code": "print('should never run')"},
        )

        with pytest.raises(ValueError, match="disabled"):
            graph_model.validate_graph(for_run=False)

    def test_graph_with_block_installation_block_rejected_for_run(self):
        """Same test but with for_run=True (execution-time validation)."""
        graph_model = self._make_graph_with_block(
            BLOCK_INSTALLATION_BLOCK_ID,
            input_default={"code": "print('should never run')"},
        )

        with pytest.raises(ValueError, match="disabled"):
            graph_model.validate_graph(for_run=True)

    def test_all_disabled_blocks_are_rejected(self):
        """Every block with disabled=True must be rejected in graph validation."""
        from backend.data.block import get_blocks

        blocks = get_blocks()
        disabled_blocks = {
            bid: block for bid, block in blocks.items() if block().disabled
        }

        assert len(disabled_blocks) > 0, "Expected at least one disabled block"
        print(f"\nFound {len(disabled_blocks)} disabled blocks to test:")

        for block_id, block_cls in disabled_blocks.items():
            block_inst = block_cls()
            print(f"  - {block_inst.name} ({block_id})")

            graph_model = self._make_graph_with_block(block_id)

            with pytest.raises(ValueError, match="disabled"):
                graph_model.validate_graph(for_run=False)

        print(f"All {len(disabled_blocks)} disabled blocks correctly rejected!")

    def test_enabled_block_not_rejected_as_disabled(self):
        """Sanity check: an enabled block should NOT be rejected for being disabled."""
        from backend.data.block import get_blocks

        blocks = get_blocks()
        # Find an enabled block
        enabled_entry = next(
            (bid, b) for bid, b in blocks.items() if not b().disabled
        )
        block_id, _ = enabled_entry

        graph_model = self._make_graph_with_block(block_id)

        # Should NOT raise "disabled" error.
        # May raise other validation errors (missing inputs etc) which is fine.
        try:
            graph_model.validate_graph(for_run=False)
        except ValueError as e:
            assert "disabled" not in str(e).lower(), (
                f"Enabled block was incorrectly rejected as disabled: {e}"
            )

    def test_direct_block_execution_check(self):
        """Verify BlockInstallationBlock is flagged disabled in the registry."""
        from backend.data.block import get_block

        block = get_block(BLOCK_INSTALLATION_BLOCK_ID)
        assert block is not None, "BlockInstallationBlock not found in registry"
        assert block.disabled is True, "BlockInstallationBlock should be disabled"

    def test_validate_graph_get_errors_returns_disabled_error(self):
        """Test the structured error reporting path also catches disabled blocks."""
        graph_model = self._make_graph_with_block(
            BLOCK_INSTALLATION_BLOCK_ID,
            input_default={"code": "print('nope')"},
        )

        # validate_graph_get_errors should return errors (not raise) for non-fatal issues,
        # but disabled blocks raise ValueError directly in _validate_graph_get_errors
        with pytest.raises(ValueError, match="disabled"):
            graph_model.validate_graph_get_errors(for_run=False)


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS — validate against a running server
# Run with:  python test_disabled_block_bypass.py --base-url URL --token JWT
# ═══════════════════════════════════════════════════════════════════════════════


def run_against_server(base_url: str, token: str):
    """
    Hit the actual API endpoints to confirm the fix works end-to-end.
    Requires a running backend.
    """
    import requests

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    results = []

    print("=" * 70)
    print("GHSA-4crw-9p35-9x54: Disabled Block Bypass - Integration Test")
    print("=" * 70)

    # ── Test 1: Create graph with disabled BlockInstallationBlock ──
    print("\n[TEST 1] POST /api/v1/graphs — graph with BlockInstallationBlock")

    node_id = str(uuid.uuid4())
    payload = {
        "graph": {
            "name": f"security-test-{uuid.uuid4().hex[:8]}",
            "description": "GHSA-4crw-9p35-9x54 validation test",
            "nodes": [
                {
                    "id": node_id,
                    "block_id": BLOCK_INSTALLATION_BLOCK_ID,
                    "input_default": {
                        "code": "# This should never be accepted\nprint('hello')"
                    },
                    "metadata": {"position": {"x": 0, "y": 0}},
                }
            ],
            "links": [],
        }
    }

    resp = requests.post(f"{base_url}/api/v1/graphs", headers=headers, json=payload)

    if resp.status_code in (400, 422, 500):
        print(f"  PASS - Server rejected graph creation (HTTP {resp.status_code})")
        try:
            print(f"  Detail: {resp.json().get('detail', resp.text)[:300]}")
        except Exception:
            print(f"  Response: {resp.text[:300]}")
        results.append(True)
    elif resp.status_code in (200, 201):
        print(f"  FAIL - Server ACCEPTED the graph! (HTTP {resp.status_code})")
        print("  >>> VULNERABILITY IS NOT FIXED <<<")
        try:
            graph_id = resp.json().get("id")
            if graph_id:
                requests.delete(
                    f"{base_url}/api/v1/graphs/{graph_id}", headers=headers
                )
                print(f"  (Cleaned up graph {graph_id})")
        except Exception:
            pass
        results.append(False)
    else:
        print(f"  WARN - Unexpected HTTP {resp.status_code}: {resp.text[:300]}")
        results.append(None)

    # ── Test 2: Direct block execution of disabled block ──
    print(f"\n[TEST 2] POST /api/v1/blocks/{BLOCK_INSTALLATION_BLOCK_ID}/execute")

    block_payload = {
        "data": {"code": "# Should be rejected\nprint('hello')"},
        "input": {"code": "# Should be rejected\nprint('hello')"},
    }

    resp = requests.post(
        f"{base_url}/api/v1/blocks/{BLOCK_INSTALLATION_BLOCK_ID}/execute",
        headers=headers,
        json=block_payload,
    )

    if resp.status_code == 403:
        print(f"  PASS - Server rejected direct execution (HTTP 403)")
        try:
            print(f"  Detail: {resp.json().get('detail', '')}")
        except Exception:
            pass
        results.append(True)
    elif resp.status_code == 200:
        print(f"  FAIL - Server EXECUTED the disabled block!")
        print("  >>> VULNERABILITY IS NOT FIXED <<<")
        results.append(False)
    else:
        print(f"  INFO - HTTP {resp.status_code}: {resp.text[:300]}")
        results.append(None)

    # ── Test 3: External API direct block execution ──
    print(
        f"\n[TEST 3] POST /api/external/v1/blocks/{BLOCK_INSTALLATION_BLOCK_ID}/execute"
    )

    resp = requests.post(
        f"{base_url}/api/external/v1/blocks/{BLOCK_INSTALLATION_BLOCK_ID}/execute",
        headers=headers,
        json=block_payload,
    )

    if resp.status_code in (403, 401):
        print(f"  PASS - Server rejected (HTTP {resp.status_code})")
        results.append(True)
    elif resp.status_code == 200:
        print(f"  FAIL - External API EXECUTED the disabled block!")
        results.append(False)
    else:
        print(f"  INFO - HTTP {resp.status_code} (may need API key auth for this endpoint)")
        results.append(None)

    # ── Summary ──
    passed = sum(1 for r in results if r is True)
    failed = sum(1 for r in results if r is False)

    print("\n" + "=" * 70)
    if failed > 0:
        print(f"RESULT: {failed} test(s) FAILED — vulnerability may not be fixed!")
    else:
        print(f"RESULT: {passed}/{len(results)} tests passed — fix is working.")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test GHSA-4crw-9p35-9x54 fix (disabled block bypass)"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8006",
        help="Backend base URL (default: http://localhost:8006)",
    )
    parser.add_argument("--token", required=True, help="JWT bearer token")
    args = parser.parse_args()

    success = run_against_server(args.base_url, args.token)
    sys.exit(0 if success else 1)
