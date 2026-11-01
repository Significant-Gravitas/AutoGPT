import { beforeAll, afterAll, afterEach, beforeEach, vi } from "vitest";
import { server } from "@/mocks/mock-server";
import { mockNextjsModules } from "./setup-nextjs-mocks";
import { mockAuthRequest } from "./mock-auth-request";
import { cleanup, configure } from "@testing-library/react";
import { toast } from "sonner";
import { resetCheckoutRedirect } from "@/hooks/useCheckoutRedirect";
import { seedFaker } from "./faker-seed";

// Auth tests send the client IP in X-Forwarded-For, as a trusted proxy would
// (lib/auth/client-ip.ts).
process.env.TRUST_X_FORWARDED_FOR ??= "true";

// Sign-ups and auth emails look up their domain's mail server
// (lib/auth/email-deliverability.ts); no test does that on the network. Every
// domain answers with no MX but an A record, so it takes mail.
vi.mock("node:dns/promises", async (importActual) => {
  const actual = await importActual<typeof import("node:dns/promises")>();
  class Resolver {
    async resolveMx() {
      return [];
    }
    async resolve4() {
      return ["192.0.2.1"];
    }
    async resolve6() {
      return [];
    }
  }
  return { ...actual, default: { ...actual, Resolver }, Resolver };
});

// React 18.3 only ships `cache` under the `react-server` export condition.
// Vitest doesn't set that condition, so any module that imports `cache` from
// "react" (e.g. server-only helpers wrapped for per-request memoization) blows
// up with "cache is not a function" the moment its file is evaluated. Shim it
// to identity here — the cache contract degrades to "no deduplication," which
// is the correct semantics in a unit-test context.
vi.mock("react", async (importActual) => {
  const actual = await importActual<typeof import("react")>();
  return {
    ...actual,
    cache: <T extends (...args: unknown[]) => unknown>(fn: T): T => fn,
  };
});

// happy-dom aborts a request still open when its file's window is torn down
// and logs the abort, which Vitest then fails as a late console write.
const openRequests = new Set<Promise<unknown>>();
const QUIET_TICKS = 5;
const MAX_SETTLE_TICKS = 1000;
const realSetTimeout = globalThis.setTimeout;

// Setup files run before each test file loads, so this seeds data built at
// module scope, in describe bodies and in beforeAll; beforeEach re-seeds.
seedFaker();

// CI runners ran these tests 2-3x slower than a dev machine (T494.1), so give
// findBy/waitFor 3x Testing Library's 1 s default.
configure({ asyncUtilTimeout: 3000 });

beforeAll(() => {
  mockNextjsModules();
  mockAuthRequest(); // If you need user's data - please mock auth actions in your specific test - it sends null user [It's only to avoid cookies() call]
  server.listen({ onUnhandledRequest: "error" });
  // After listen(): MSW's fetch wrapper answers mocked requests without
  // calling the fetch it wrapped, so a tracker beneath it sees none.
  trackOpenRequests();
});
beforeEach(() => {
  seedFaker();
});
afterEach(async () => {
  cleanup();
  await settleOpenRequests();
  server.resetHandlers();
  // The checkout redirect is page-wide state that outlives a render; a
  // redirect left by one test would disable the next test's buttons.
  resetCheckoutRedirect();
  // Sonner keeps toasts in module state and replays the ones still showing
  // to every Toaster that mounts, so a toast left by one test would show up
  // again in the next test's Toaster.
  toast.dismiss();
});
afterAll(() => server.close());

function trackOpenRequests() {
  const fetch = globalThis.fetch;
  globalThis.fetch = (...args: Parameters<typeof fetch>) => {
    const request = fetch(...args);
    openRequests.add(request);
    request.then(
      () => openRequests.delete(request),
      () => openRequests.delete(request),
    );
    return request;
  };
}

// cleanup() runs pending mount effects whose requests start a few ticks later,
// so wait for a quiet spell; bounded, as some tests hold a request open.
async function settleOpenRequests() {
  let quiet = 0;
  for (let tick = 0; tick < MAX_SETTLE_TICKS && quiet < QUIET_TICKS; tick++) {
    await new Promise((resolve) => realSetTimeout(resolve, 0));
    quiet = openRequests.size === 0 ? quiet + 1 : 0;
  }
  openRequests.clear();
}
