import { SentryUserTracker } from "@/components/monitor/SentryUserTracker";
import * as Sentry from "@sentry/nextjs";
import { cleanup, render } from "@testing-library/react";
import {
  afterAll,
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";

// The client bundle runs the browser SDK; Vitest would load the Node one.
vi.mock("@sentry/nextjs", async () => {
  const { createRequire } = await import("node:module");
  const nextjs = createRequire(import.meta.url).resolve(
    "@sentry/nextjs/package.json",
  );
  const react = createRequire(nextjs).resolve("@sentry/react/package.json");
  const browser = createRequire(react).resolve("@sentry/browser/package.json");
  const sdk = await import(
    browser.replace("package.json", "build/npm/esm/prod/index.js")
  );
  return { ...sdk, captureRouterTransitionStart: () => undefined };
});

const EMAIL = "ada@example.com";
const IP = "203.0.113.7";

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "user-1", email: EMAIL, role: "authenticated" },
    isUserLoading: false,
  }),
}));

type SentPayload = Sentry.Event & {
  sdk?: { settings?: { infer_ip?: string } };
};

// Errors and transactions as the production client config would send them.
const sent: SentPayload[] = [];

beforeAll(async () => {
  vi.spyOn(console, "info").mockImplementation(() => undefined);
  vi.stubEnv("NEXT_PUBLIC_SENTRY_DSN", "https://public@o0.ingest.sentry.io/0");
  await import("../../instrumentation-client");
  vi.spyOn(Sentry.getClient()!.getTransport()!, "send").mockImplementation(
    async ([, items]) => {
      for (const [header, payload] of items) {
        if (header.type === "event" || header.type === "transaction") {
          sent.push(payload as SentPayload);
        }
      }
      return {};
    },
  );
});

beforeEach(() => {
  sent.length = 0;
});

afterEach(() => {
  cleanup();
  Sentry.setUser(null);
});

afterAll(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("a signed-in user in Sentry", () => {
  it("is identified by id and role only, with no email or IP address", async () => {
    render(<SentryUserTracker />);

    Sentry.captureException(new Error("boom"));
    Sentry.startSpan({ name: "load page", forceTransaction: true }, () => {});
    await Sentry.flush(2000);

    expect(sent.map((payload) => payload.type ?? "error")).toEqual([
      "error",
      "transaction",
    ]);
    for (const payload of sent) {
      expect(payload.user).toEqual({ id: "user-1", role: "authenticated" });
      // Without it Sentry would fill in the IP the envelope came from.
      expect(payload.sdk?.settings?.infer_ip).toBe("never");
      expect(JSON.stringify(payload)).not.toContain(EMAIL);
    }
  });

  it("keeps only the id and role of a user other code puts on the scope", async () => {
    Sentry.setUser({
      id: "user-1",
      role: "admin",
      email: EMAIL,
      ip_address: IP,
      username: "ada",
      name: "Ada Lovelace",
    });

    Sentry.captureException(new Error("boom"));
    Sentry.startSpan({ name: "load page", forceTransaction: true }, () => {});
    await Sentry.flush(2000);

    expect(sent).toHaveLength(2);
    for (const payload of sent) {
      expect(payload.user).toEqual({ id: "user-1", role: "admin" });
      expect(JSON.stringify(payload)).not.toContain(EMAIL);
      expect(JSON.stringify(payload)).not.toContain(IP);
      expect(JSON.stringify(payload)).not.toContain("Lovelace");
    }
  });
});
