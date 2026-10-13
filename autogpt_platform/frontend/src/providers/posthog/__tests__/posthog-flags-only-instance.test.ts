import { server } from "@/mocks/mock-server";
import { waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Through the real posthog-js: dev reads prod's flags with this instance, so
// the evidence that it leaves no event, person or storage behind in prod has
// to come from the SDK itself, not from a mock of it.

const FLAGS_HOST = "https://flags-project.posthog.test";
// posthog-js drops every event from what looks like a bot (happy-dom reports a
// webdriver), which would make "nothing was captured" true for the wrong reason.
const BROWSER_USER_AGENT =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36";

interface SentRequest {
  method: string;
  path: string;
  body: Record<string, unknown> | null;
}

describe("the flags-only PostHog instance", () => {
  beforeEach(() => {
    vi.stubEnv("NEXT_PUBLIC_APP_ENV", "dev");
    vi.spyOn(window.navigator, "userAgent", "get").mockReturnValue(
      BROWSER_USER_AGENT,
    );
    vi.spyOn(window.navigator, "webdriver", "get").mockReturnValue(false);
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  it("reads flags from the flags project and sends it nothing else", async () => {
    const requests = recordFlagsProject();
    const { getFlagsClient, identifyFlagsOnlyClient, initFlagsOnlyClient } =
      await import("../posthog-flags-client");

    initFlagsOnlyClient({
      flagsKey: "phc_prod",
      analyticsKey: "phc_dev",
      hosts: { api_host: FLAGS_HOST },
      anonymousID: "anon-1",
    });
    const client = getFlagsClient();
    const captured = vi.fn();
    client.on("eventCaptured", captured);

    await waitFor(() => expect(client.featureFlags.hasLoadedFlags).toBe(true));
    expect(client.isFeatureEnabled("hire-experts")).toBe(true);
    expect(client.getFeatureFlag("pricing-test")).toBe("yearly-pro");
    identifyFlagsOnlyClient("user-1");
    expect(client.capture("$pageview")).toBeUndefined();
    await waitFor(() => expect(flagsCalls(requests)).toHaveLength(2));

    const [first, afterIdentify] = flagsCalls(requests);
    expect(first.body).toMatchObject({
      token: "phc_prod",
      distinct_id: "anon-1",
      person_properties: { environment: "dev" },
    });
    expect(afterIdentify.body).toMatchObject({
      token: "phc_prod",
      distinct_id: "user-1",
      person_properties: { environment: "dev" },
    });
    // $feature_flag_called, $identify and $pageview all stop at before_send.
    expect(captured).not.toHaveBeenCalled();
    expect(
      requests.filter(
        (request) =>
          request.path !== "/flags/" && !request.path.startsWith("/array/"),
      ),
    ).toEqual([]);
    expect(storedByPostHog()).toEqual([]);
  });
});

function recordFlagsProject() {
  const requests: SentRequest[] = [];
  server.use(
    http.all(`${FLAGS_HOST}/*`, async ({ request }) => {
      const path = new URL(request.url).pathname;
      requests.push({
        method: request.method,
        path,
        body: decodeBody(await request.text()),
      });
      if (path !== "/flags/") return new HttpResponse(null, { status: 404 });
      return HttpResponse.json({
        featureFlags: { "hire-experts": true, "pricing-test": "yearly-pro" },
        featureFlagPayloads: {},
      });
    }),
  );
  return requests;
}

function flagsCalls(requests: SentRequest[]) {
  return requests.filter((request) => request.path === "/flags/");
}

// /flags goes out as `data=<base64 JSON>`.
function decodeBody(body: string) {
  const data = new URLSearchParams(body).get("data");
  if (!data) return null;
  return JSON.parse(atob(data)) as Record<string, unknown>;
}

function storedByPostHog() {
  const keys = [
    ...Object.keys(window.localStorage),
    ...Object.keys(window.sessionStorage),
    ...document.cookie.split(";").map((part) => part.trim().split("=")[0]),
  ];
  return keys.filter((key) => key.includes("ph_") || key.includes("posthog"));
}
