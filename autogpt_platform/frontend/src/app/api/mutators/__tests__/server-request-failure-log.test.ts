// @vitest-environment node
import { getV2GetLibraryAgent } from "@/app/api/__generated__/endpoints/library/library";
import { server } from "@/mocks/mock-server";
import {
  hangUp,
  startLoopbackBackend,
} from "@/tests/integrations/loopback-backend";
import * as Sentry from "@sentry/nextjs";
import { http, HttpResponse } from "msw";
import {
  afterAll,
  afterEach,
  beforeAll,
  describe,
  expect,
  it,
  vi,
} from "vitest";

// A signed-in render; reading the session cookie needs a live Next request.
vi.mock("@/lib/autogpt-server-api/helpers", async (importActual) => ({
  ...(await importActual<typeof import("@/lib/autogpt-server-api/helpers")>()),
  getServerAuthToken: async () => "session-token",
}));

// A real server-side Sentry client: its console integration turns every log
// line into a breadcrumb on the next server event, which is where a logged
// object used to read "[object Object]".
const sent: Sentry.ErrorEvent[] = [];
// Spied before init, so Sentry's console patch wraps the spies.
const consoleError = vi
  .spyOn(console, "error")
  .mockImplementation(() => undefined);
const consoleInfo = vi
  .spyOn(console, "info")
  .mockImplementation(() => undefined);

beforeAll(() => {
  vi.spyOn(console, "warn").mockImplementation(() => undefined);
  Sentry.init({
    dsn: "https://examplePublicKey@o0.ingest.sentry.io/0",
    transport: () => ({
      send: () => Promise.resolve({}),
      flush: () => Promise.resolve(true),
    }),
    beforeSend(event) {
      sent.push(event);
      return null;
    },
  });
});

afterAll(() => {
  vi.restoreAllMocks();
});

const AGENT = "9c1f6a52-3c0e-4b7e-9d0a-6f1e2b3c4d5e";

function failWith(status: number, detail: string) {
  server.use(
    http.get("*/api/library/agents/:id", () =>
      HttpResponse.json({ detail }, { status }),
    ),
  );
}

describe("a request that fails while rendering on the server", () => {
  it("logs one readable line, and Sentry's breadcrumb reads the same", async () => {
    failWith(502, "Backend sent no response within 30000ms");

    await expect(getV2GetLibraryAgent(AGENT)).rejects.toThrow();
    Sentry.captureMessage("Server render failed");
    await Sentry.flush(2000);

    const line =
      "Request failed on server: GET /api/library/agents/{id} 502: Backend sent no response within 30000ms";
    expect(consoleError).toHaveBeenCalledExactlyOnceWith(line);
    const crumbs = sent[0].breadcrumbs?.filter((b) => b.category === "console");
    expect(crumbs?.map((b) => b.message)).toContain(line);
    expect(JSON.stringify(crumbs)).not.toContain("[object Object]");
  });

  it("logs an outcome the app handles, like a deleted agent, as info", async () => {
    consoleError.mockClear();
    failWith(404, "Library agent not found");

    await expect(getV2GetLibraryAgent(AGENT)).rejects.toThrow();

    expect(consoleError).not.toHaveBeenCalled();
    expect(consoleInfo).toHaveBeenCalledWith(
      "Request failed on server: GET /api/library/agents/{id} 404: Library agent not found",
    );
  });
});

// During a server render, Next's fetch gives an identical GET that has no
// signal the first call's promise instead of sending it again
// (next/dist/server/lib/dedupe-fetch.js); a signal opts out. This keeps those
// rules, so a resend that doesn't opt out gets the first failure back.
function memoizeLikeANextRender(realFetch: typeof fetch): typeof fetch {
  const sent = new Map<string, Promise<Response>>();
  return (input, init) => {
    const method = (init?.method ?? "GET").toUpperCase();
    if (init?.signal || (method !== "GET" && method !== "HEAD")) {
      return realFetch(input, init);
    }
    const url = input instanceof Request ? input.url : String(input);
    const first = sent.get(url);
    if (first) return first.then((response) => response.clone());
    const promise = realFetch(input, init);
    sent.set(url, promise);
    return promise.then((response) => response.clone());
  };
}

describe("a read whose backend connection drops while rendering on the server", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it("is sent again even though the render reuses identical reads", async () => {
    const backend = await startLoopbackBackend((request, response, count) => {
      if (count === 1) return hangUp(request);
      response.setHeader("Content-Type", "application/json");
      response.end(JSON.stringify({ id: AGENT }));
    });
    vi.stubEnv("AGPT_SERVER_URL", `${backend.url}/api`);
    vi.stubGlobal("fetch", memoizeLikeANextRender(globalThis.fetch));

    const response = await getV2GetLibraryAgent(AGENT);

    expect(response.status).toBe(200);
    expect(backend.requests).toHaveLength(2);
  });

  it("is sent once more on a new connection, and only logged at info", async () => {
    consoleError.mockClear();
    consoleInfo.mockClear();
    const backend = await startLoopbackBackend((request, response, count) => {
      if (count === 1) return hangUp(request);
      response.setHeader("Content-Type", "application/json");
      response.end(JSON.stringify({ id: AGENT }));
    });
    vi.stubEnv("AGPT_SERVER_URL", `${backend.url}/api`);

    const response = await getV2GetLibraryAgent(AGENT);

    expect(response.status).toBe(200);
    expect(backend.requests).toEqual([
      `GET /api/library/agents/${AGENT}`,
      `GET /api/library/agents/${AGENT}`,
    ]);
    expect(consoleInfo).toHaveBeenCalledExactlyOnceWith(
      "Retried GET /api/library/agents/{id} after a dropped connection (UND_ERR_SOCKET): 200",
    );
    expect(consoleError).not.toHaveBeenCalled();
  });
});
