// Cancels the Stripe test-mode subscriptions that load-test runs left
// behind, which happens when a run stops before its teardown or its setup()
// runs out of time. They are found by the autogpt_load_test_run metadata
// the runs put on them: every run's by default, or one run's with
// RUN_FILTER. Needs STRIPE_TEST_SECRET_KEY, as the runs do (README:
// Paywalled targets).
//
//   ./k6.sh run scripts/cancel-load-test-subscriptions.js
//   RUN_FILTER=20261015T104309Z-1234 ./k6.sh run scripts/cancel-load-test-subscriptions.js
import {
  cancelSubscriptions,
  listRunSubscriptions,
  stripeTestKey,
} from "../lib/stripe.js";

export const options = {
  scenarios: {
    cancel: {
      executor: "shared-iterations",
      vus: 1,
      iterations: 1,
      maxDuration: "10m",
    },
  },
  userAgent: "autogpt-load-tests/2 (cancel-load-test-subscriptions)",
};

export default function () {
  if (!stripeTestKey(null)) {
    throw new Error(
      "STRIPE_TEST_SECRET_KEY is not set; nothing to cancel with",
    );
  }
  const runId = __ENV.RUN_FILTER || "";
  const found = listRunSubscriptions(runId);
  console.log(
    `found ${found.length} active load-test subscription(s)${runId ? ` of run ${runId}` : ""}`,
  );
  if (found.length === 0) return;
  const result = cancelSubscriptions(found.map((s) => s.id));
  console.log(`cancelled ${result.cancelled}, failed ${result.failed}`);
}
