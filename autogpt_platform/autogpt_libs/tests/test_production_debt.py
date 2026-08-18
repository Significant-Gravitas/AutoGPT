import importlib.util
import os
import sys
import unittest

# Load module directly
file_path = os.path.join(
    os.path.dirname(__file__),
    "../autogpt_libs/production_debt.py",
)
spec = importlib.util.spec_from_file_location("autogpt_production_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["autogpt_production_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtInterceptor = production_debt_mod.ProductionDebtInterceptor
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtInterceptor(unittest.TestCase):
    def setUp(self) -> None:
        self.interceptor = ProductionDebtInterceptor(
            never_equate_intent_to_approval=True,
            max_acceptable_adi=12.0,
        )

    def test_clean_agent_step_passes_readiness(self) -> None:
        report = self.interceptor.intercept_step(
            agent_id="agent_alpha_01",
            step_index=1,
            context_tokens=1000,
            generated_tokens=100,
            step_latency_seconds=0.85,
            recursive_loop_count=0,
            un_gated_mutations=0,
        )
        self.assertTrue(report.is_production_ready)
        self.assertLessEqual(report.adi_score, 12.0)
        self.assertEqual(len(report.critical_smells), 0)
        self.assertTrue(bool(report.receipt_hash))

    def test_degraded_agent_step_fails_debt(self) -> None:
        report = self.interceptor.intercept_step(
            agent_id="agent_runaway_99",
            step_index=15,
            context_tokens=1000,
            generated_tokens=3000,  # High token sprawl (4.0x)
            step_latency_seconds=7.5,  # High latency
            recursive_loop_count=5,  # 5 loops
            un_gated_mutations=2,  # 2 un-gated command mutations
        )
        self.assertFalse(report.is_production_ready)
        self.assertGreater(report.adi_score, 50.0)
        self.assertIn("HIGH_TOKEN_SPRAWL_4.00X", report.critical_smells)
        self.assertIn("HIGH_STEP_LATENCY_7.50S", report.critical_smells)
        self.assertIn("DETECTED_5_RECURSIVE_THOUGHT_LOOPS", report.critical_smells)
        self.assertIn("DETECTED_2_UNGATED_MUTATIONS", report.critical_smells)

    def test_cryptographic_ledger_integrity(self) -> None:
        self.interceptor.intercept_step("ag-1")
        self.interceptor.intercept_step("ag-2")
        self.interceptor.intercept_step("ag-3")

        entries = self.interceptor.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.interceptor.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
