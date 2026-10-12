import { listCopilotFollowupSchedules } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { resetClientRequestFailureLog } from "@/app/api/mutators/request-failure-log";
import { server } from "@/mocks/mock-server";
import * as Sentry from "@sentry/nextjs";
import { http, HttpResponse } from "msw";
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

// Vitest resolves @sentry/nextjs to its Node build, which has no HttpClient
// integration; the client bundle runs the browser SDK, so load that one.
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

// Events as the production client config would send them.
const sent: Sentry.ErrorEvent[] = [];

beforeAll(async () => {
  vi.spyOn(console, "info").mockImplementation(() => undefined);
  vi.stubEnv("NEXT_PUBLIC_SENTRY_DSN", "https://public@o0.ingest.sentry.io/0");
  // HttpClient only hooks a fetch it takes for the browser's own, which MSW's
  // is not; Sentry trusts any fetch on an edge runtime.
  vi.stubGlobal("EdgeRuntime", "vitest");
  await import("../../instrumentation-client");
  vi.spyOn(Sentry.getClient()!.getTransport()!, "send").mockImplementation(
    async ([, items]) => {
      for (const [header, payload] of items) {
        if (header.type === "event") sent.push(payload as Sentry.ErrorEvent);
      }
      return {};
    },
  );
});

beforeEach(() => {
  sent.length = 0;
});

afterEach(() => {
  resetClientRequestFailureLog();
});

afterAll(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const FILE = "6f1e2b3c-4d5e-4b7e-9d0a-9c1f6a523c0e";

describe("a failed API request in Sentry", () => {
  it("is reported once, under its route, when the generated client made it", async () => {
    server.use(
      http.get("*/api/proxy/api/schedules/followups", () =>
        HttpResponse.json(
          { detail: "All connection attempts failed" },
          { status: 500 },
        ),
      ),
    );

    await expect(listCopilotFollowupSchedules()).rejects.toThrow(
      "All connection attempts failed",
    );
    await Sentry.flush(2000);

    expect(sent.map((event) => event.message)).toEqual([
      "Request failed on client: GET /api/schedules/followups 500",
    ]);
  });

  it("is still reported once when the request bypassed the generated client", async () => {
    server.use(
      http.get("*/api/proxy/api/workspace/files/:id/download", () =>
        HttpResponse.text("upstream unavailable", { status: 503 }),
      ),
    );

    const response = await fetch(
      `/api/proxy/api/workspace/files/${FILE}/download`,
    );
    await Sentry.flush(2000);

    expect(response.status).toBe(503);
    expect(sent.map((event) => event.message)).toEqual([
      "HTTP Client Error with status code: 503",
    ]);
  });
});
