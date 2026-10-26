import * as Sentry from "@sentry/nextjs";
import {
  afterAll,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";

// The client bundle runs the browser SDK; Vitest would load the Node one.
vi.mock("@sentry/nextjs", async () => {
  const sdk = await import(/* @vite-ignore */ await browserSDKEntry());
  return { ...sdk, captureRouterTransitionStart: () => undefined };
});

const R = "[REDACTED]";
const RESET_MARKER = "marker-r-4f1c";
const LINK_MARKER = "marker-l-9a2e";
const RESET_PAGE = `/reset-password?token=${RESET_MARKER}`;
const LINK_PAGE = `/link/${LINK_MARKER}?platform=discord`;
const LINK_INFO_CALL = `/api/proxy/api/platform-linking/tokens/${LINK_MARKER}/info`;

const sent: Sentry.Event[] = [];
const sentSpans: { data?: Record<string, unknown> }[] = [];

beforeAll(async () => {
  vi.spyOn(console, "info").mockImplementation(() => undefined);
  vi.stubEnv("NEXT_PUBLIC_SENTRY_DSN", "https://public@o0.ingest.sentry.io/0");
  await import("../../instrumentation-client");
  vi.spyOn(Sentry.getClient()!.getTransport()!, "send").mockImplementation(
    async ([, items]) => {
      for (const [header, payload] of items) {
        if (header.type === "event" || header.type === "transaction") {
          sent.push(payload as Sentry.Event);
        }
        if (header.type === "span") sentSpans.push(payload as never);
      }
      return {};
    },
  );
});

beforeEach(() => {
  sent.length = 0;
  sentSpans.length = 0;
  Sentry.getIsolationScope().clearBreadcrumbs();
  Sentry.getCurrentScope().setTransactionName(undefined);
  vi.spyOn(document, "referrer", "get").mockReturnValue("");
});

afterAll(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("the browser client", () => {
  it("records no session replay", () => {
    const client = Sentry.getClient()!;
    expect(client.getIntegrationByName("Replay")).toBeUndefined();
    expect(Sentry.getReplay()).toBeUndefined();
  });
});

describe("a browser error report", () => {
  it("masks the token in the page's address, its referrer and the navigation to it", async () => {
    vi.spyOn(document, "referrer", "get").mockReturnValue(
      `${location.origin}${RESET_PAGE}`,
    );
    history.replaceState({}, "", RESET_PAGE);
    history.pushState({}, "", LINK_PAGE);
    Sentry.getCurrentScope().setTransactionName(`/link/${LINK_MARKER}`);
    Sentry.addBreadcrumb({
      category: "fetch",
      type: "http",
      data: { method: "GET", url: LINK_INFO_CALL, status_code: 200 },
    });

    Sentry.captureException(new Error("boom"));
    await Sentry.flush(2000);

    expect(sent).toHaveLength(1);
    const [event] = sent;
    expect(event.request?.url).toBe(
      `${location.origin}/link/${R}?platform=${R}`,
    );
    expect(event.request?.headers?.Referer).toBe(
      `${location.origin}/reset-password?token=${R}`,
    );
    expect(event.transaction).toBe(`/link/${R}`);
    const crumbs = event.breadcrumbs?.map((crumb) => crumb.data);
    expect(crumbs).toContainEqual({
      from: `/reset-password?token=${R}`,
      to: `/link/${R}?platform=${R}`,
    });
    expect(crumbs).toContainEqual({
      method: "GET",
      url: `/api/proxy/api/platform-linking/tokens/${R}/info`,
      status_code: 200,
    });
    expect(JSON.stringify(event)).not.toContain(RESET_MARKER);
    expect(JSON.stringify(event)).not.toContain(LINK_MARKER);
  });

  it("keeps an ordinary page's address as it is", async () => {
    history.replaceState({}, "", "/library/agents/a-1");
    history.pushState({}, "", "/marketplace");

    Sentry.captureException(new Error("boom"));
    await Sentry.flush(2000);

    expect(sent).toHaveLength(1);
    const [event] = sent;
    expect(event.request?.url).toBe(`${location.origin}/marketplace`);
    expect(event.breadcrumbs?.map((crumb) => crumb.data)).toContainEqual({
      from: "/library/agents/a-1",
      to: "/marketplace",
    });
  });
});

describe("a browser transaction", () => {
  it("masks the token in its name, its page and its spans", async () => {
    history.replaceState({}, "", LINK_PAGE);

    Sentry.startSpan(
      {
        name: `/link/${LINK_MARKER}`,
        forceTransaction: true,
        attributes: { "sentry.source": "url" },
      },
      () => {
        Sentry.startSpan(
          {
            name: `GET ${LINK_INFO_CALL}`,
            op: "http.client",
            attributes: {
              url: LINK_INFO_CALL,
              "http.query": `?token=${RESET_MARKER}`,
              "http.method": "GET",
            },
          },
          () => {},
        );
      },
    );
    await Sentry.flush(2000);

    expect(sent).toHaveLength(1);
    const [transaction] = sent;
    expect(transaction.transaction).toBe(`/link/${R}`);
    expect(transaction.request?.url).toBe(
      `${location.origin}/link/${R}?platform=${R}`,
    );
    const [span] = transaction.spans ?? [];
    expect(span.description).toBe(
      `GET /api/proxy/api/platform-linking/tokens/${R}/info`,
    );
    expect(span.data).toMatchObject({
      url: `/api/proxy/api/platform-linking/tokens/${R}/info`,
      "http.query": `?token=${R}`,
      "http.method": "GET",
    });
    expect(JSON.stringify(transaction)).not.toContain(RESET_MARKER);
    expect(JSON.stringify(transaction)).not.toContain(LINK_MARKER);
  });

  it("sends web vitals with the page they were measured on masked", async () => {
    Sentry.startInactiveSpan({
      name: "interaction",
      experimental: { standalone: true },
      attributes: { transaction: `/link/${LINK_MARKER}` },
    }).end();
    await Sentry.flush(2000);

    expect(sentSpans).toHaveLength(1);
    expect(sentSpans[0].data?.transaction).toBe(`/link/${R}`);
  });
});

// The browser SDK's build, resolved as pnpm laid it out.
async function browserSDKEntry() {
  const { createRequire } = await import("node:module");
  const nextjs = createRequire(import.meta.url).resolve(
    "@sentry/nextjs/package.json",
  );
  const react = createRequire(nextjs).resolve("@sentry/react/package.json");
  const browser = createRequire(react).resolve("@sentry/browser/package.json");
  return browser.replace("package.json", "build/npm/esm/prod/index.js");
}
