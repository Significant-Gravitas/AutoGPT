import type { User } from "@/lib/auth/types";
import { server } from "@/mocks/mock-server";
import {
  answerCookiebot,
  configureCookiebot,
  installCookiebot,
  removeCookiebot,
} from "@/tests/integrations/cookiebot";
import { act, render } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { PostHog, type CaptureResult } from "posthog-js";
import { stripStoredURLCredentials } from "../posthog-stored-urls";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Runs the real posthog-js client behind our provider against a stubbed
// PostHog host, so each event is read as posthog-js hands it to its request
// queue, after the provider's before_send chain.

const POSTHOG_ORIGIN = "https://posthog.test";
const PAGE_ORIGIN = "http://localhost:3000";
const LINK_VALUE = "one-time-link-value";
const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "*",
  "Access-Control-Allow-Methods": "*",
};

let pageLoads = 0;
let captured: CaptureResult[] = [];
let flagsBodies: FlagsBody[] = [];
let clients: PostHog[] = [];
let captureQueueResets: (() => void)[] = [];

beforeEach(() => {
  vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "CLOUD");
  vi.stubEnv("NEXT_PUBLIC_POSTHOG_HOST", POSTHOG_ORIGIN);
  vi.stubEnv("NEXT_PUBLIC_COOKIEBOT_CBID", "");
  // posthog-js tries a script tag for remote config first; happy-dom refuses.
  vi.spyOn(console, "error").mockImplementation(() => {});
  // PostHog drops events from browsers it takes for bots, HappyDOM included.
  vi.spyOn(navigator, "webdriver", "get").mockReturnValue(false);
  captured = [];
  flagsBodies = [];
  server.use(
    http.options(
      `${POSTHOG_ORIGIN}/*`,
      () => new HttpResponse(null, { status: 204, headers: CORS_HEADERS }),
    ),
    http.all(`${POSTHOG_ORIGIN}/*`, async ({ request }) => {
      if (new URL(request.url).pathname.endsWith("/flags/")) {
        flagsBodies.push(await readFlagsBody(request));
      }
      return HttpResponse.json({}, { headers: CORS_HEADERS });
    }),
  );
  configureCookiebot();
  installCookiebot({ statistics: true });
});

afterEach(() => {
  clients.forEach((client) => client.set_config({ persistence: "memory" }));
  clients = [];
  captureQueueResets.forEach((reset) => reset());
  captureQueueResets = [];
  removeCookiebot();
  Reflect.deleteProperty(document, "referrer");
  setPageURL(`${PAGE_ORIGIN}/`);
  window.localStorage.clear();
  window.sessionStorage.clear();
  document.cookie
    .split(";")
    .map((part) => part.trim().split("=")[0])
    .filter(Boolean)
    .forEach((name) => {
      document.cookie = `${name}=; Path=/; Expires=Thu, 01 Jan 1970 00:00:00 GMT`;
    });
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
});

describe("analytics events from a page whose URL carries a credential", () => {
  it("strip the link token from our pageview and keep next", async () => {
    await loadPage("/reset-password?token=SECRET&next=%2Fhome&utm_source=mail");

    const pageview = await waitForEvent("$pageview");

    // happy-dom has no window.origin, which the tracker puts in front.
    expect(pageview.properties.$current_url).toMatch(
      /\/reset-password\?token=\[REDACTED\]&next=%2Fhome&utm_source=mail$/,
    );
    expectNoSecret(pageview);
  });

  it("strip it from PostHog's own page-leave event", async () => {
    await loadPage("/reset-password?token=SECRET&next=%2Fhome");
    await waitForEvent("$pageview");

    act(() => {
      // posthog-js listens for unload where the browser has no pagehide.
      window.dispatchEvent(
        new Event("onpagehide" in self ? "pagehide" : "unload"),
      );
    });
    const pageleave = await waitForEvent("$pageleave");

    expect(pageleave.properties.$current_url).toBe(
      `${PAGE_ORIGIN}/reset-password?token=[REDACTED]&next=%2Fhome`,
    );
    expectNoSecret(pageleave);
  });

  it("strip it from the referrer the next page reports", async () => {
    Object.defineProperty(document, "referrer", {
      configurable: true,
      value: withSecret(
        `${PAGE_ORIGIN}/reset-password?token=SECRET&next=%2Fhome`,
      ),
    });
    await loadPage("/login");

    const pageview = await waitForEvent("$pageview");

    expect(pageview.properties.$referrer).toBe(
      `${PAGE_ORIGIN}/reset-password?token=[REDACTED]&next=%2Fhome`,
    );
    expectNoSecret(pageview);
  });

  it("strip it from the first page an identified person is given", async () => {
    // Identify runs on the grant; with a stored one it would precede init.
    removeCookiebot();
    installCookiebot();
    await loadPage("/settings/account?f=off&t=SECRET", { user: USER });
    act(() => answerCookiebot({ statistics: true }));

    const identify = await waitForEvent("$identify");

    expect(identify.$set_once?.$initial_current_url).toContain(
      "/settings/account?f=off&t=[REDACTED]",
    );
    expectNoSecret(identify);
  });

  it("strip it from the page keys of PostHog's heatmap data", async () => {
    const posthog = await loadPage("/reset-password?token=SECRET&next=%2Fhome");
    await waitForEvent("$pageview");
    posthog.set_config({ capture_heatmaps: true });

    act(() => {
      document.body.click();
      window.dispatchEvent(new Event("beforeunload"));
    });
    const heatmap = await waitForEvent("$$heatmap");

    expect(Object.keys(heatmap.properties.$heatmap_data)).toEqual([
      expect.stringMatching(/\?token=\[REDACTED\]&next=%2Fhome$/),
    ]);
    expectNoSecret(heatmap);
  });

  // Every page whose URL holds a credential, and a parameter that must survive.
  it.each([
    [
      "the password reset link",
      "/reset-password?token=SECRET",
      "/reset-password?token=",
    ],
    [
      "the signed email-preference link",
      "/settings/account?f=off&t=SECRET",
      "f=off",
    ],
    [
      "the OAuth authorize page",
      "/auth/authorize?client_id=app&state=SECRET&code_challenge=abc",
      "client_id=app",
    ],
    [
      "the integration setup wizard",
      "/auth/integrations/setup-wizard?client_id=app&state=SECRET",
      "client_id=app",
    ],
    [
      "an OAuth callback",
      "/auth/integrations/oauth_callback?code=SECRET&state=SECRET",
      "/oauth_callback?code=",
    ],
    [
      "the sign-in redirect from a signed link",
      "/login?next=%2Fsettings%2Faccount%3Ff%3Doff%26t%3DSECRET",
      "next=%2Fsettings%2Faccount%3Ff%3Doff%26t%3D",
    ],
    ["the chat platform linking page", "/link/SECRET", "/link/"],
    [
      "the linking page's Telegram return",
      "/link/SECRET?id=42&auth_date=1700000000&hash=SECRET",
      "auth_date=1700000000",
    ],
    ["a shared run", "/share/SECRET", "/share/"],
    ["a shared chat", "/share/chat/SECRET", "/share/chat/"],
    [
      "a token in the fragment",
      "/auth/auth-code-error#access_token=SECRET&type=recovery",
      "type=recovery",
    ],
  ])("strip it from %s", async (_page, path, kept) => {
    const posthog = await loadPage(path);

    posthog.capture("probe");
    const probe = await waitForEvent("probe");

    expect(probe.properties.$current_url).toContain(kept);
    expectNoSecret(probe);
  });
});

// posthog-js sends the first page it stored with every /flags/ request, which
// skips before_send, and keeps it in the cookie agpt.co's PostHog reads too.
describe("the first page posthog-js stores", () => {
  it("is masked before the first flag request when an earlier visit stored it", async () => {
    storeAsEarlierVisit(SIGNED_LINK_REDIRECT, LINKING_PAGE);

    await loadPage("/marketplace", { trackers: false });

    const firstPage = await waitForStoredPageInFlags();
    expect(firstPage.$initial_current_url).toContain(MASKED_NEXT);
    expect(firstPage.$initial_referrer).toBe(`${PAGE_ORIGIN}/link/[REDACTED]`);
    expectPostHogStorageHoldsNoSecret();
  });

  it.each([
    [
      "the password reset link",
      "/reset-password?token=SECRET&next=%2Fhome",
      undefined,
      "token=[REDACTED]&next=%2Fhome",
    ],
    [
      "a signed link's sign-in redirect, from the linking page",
      SIGNED_LINK_REDIRECT,
      LINKING_PAGE,
      MASKED_NEXT,
    ],
  ])(
    "is masked when this visit stores %s after the visitor accepts",
    async (_page, path, referrer, kept) => {
      removeCookiebot();
      installCookiebot();
      setReferrer(referrer);
      const posthog = await loadPage(path);
      act(() => answerCookiebot({ statistics: true }));
      await waitForEvent("$pageview");

      posthog.reloadFeatureFlags();
      const firstPage = await waitForStoredPageInFlags();

      expect(firstPage.$initial_current_url).toContain(kept);
      expectPostHogStorageHoldsNoSecret();
    },
  );

  it("is masked when an earlier visit stored it and the visitor accepts later", async () => {
    removeCookiebot();
    installCookiebot();
    storeAsEarlierVisit(SIGNED_LINK_REDIRECT, LINKING_PAGE);
    const posthog = await loadPage("/marketplace", { trackers: false });

    act(() => answerCookiebot({ statistics: true }));
    posthog.reloadFeatureFlags();
    const firstPage = await waitForStoredPageInFlags();

    expect(firstPage.$initial_current_url).toContain(MASKED_NEXT);
    expectPostHogStorageHoldsNoSecret();
  });

  it("reaches agpt.co's PostHog, which reads the shared cookie, masked", async () => {
    setReferrer(LINKING_PAGE);
    await loadPage(SIGNED_LINK_REDIRECT);
    await waitForEvent("$pageview");
    // agpt.co is another origin: only the cookie is shared.
    window.localStorage.clear();
    window.sessionStorage.clear();
    setPageURL(`${PAGE_ORIGIN}/`);
    setReferrer(undefined);
    flagsBodies = [];

    const website = new PostHog();
    clients.push(website);
    website.init(`phc_test_${pageLoads}`, {
      api_host: POSTHOG_ORIGIN,
      persistence: "localStorage+cookie",
      cross_subdomain_cookie: true,
      capture_pageview: false,
      autocapture: false,
      disable_session_recording: true,
    });
    website.reloadFeatureFlags();
    const event = website.capture("probe");

    const firstPage = await waitForStoredPageInFlags();
    expect(firstPage.$initial_current_url).toContain(MASKED_NEXT);
    expect(JSON.stringify(event)).not.toContain(LINK_VALUE);
  });

  it("is masked under any key posthog-js stores it", async () => {
    const posthog = await loadPage("/marketplace", { trackers: false });
    posthog.register({
      $renamed_initial_info: { u: withSecret(`${PAGE_ORIGIN}/link/SECRET`) },
    });

    stripStoredURLCredentials(posthog);

    expect(posthog.get_property("$renamed_initial_info")).toEqual({
      u: `${PAGE_ORIGIN}/link/[REDACTED]`,
    });
  });
});

// rrweb cannot record in happy-dom (it patches canvas and CSS prototypes it
// lacks), so the batch is built as the recorder sends it and captured through
// the same client.
describe("session replay of a page whose URL carries a credential", () => {
  it("records the page URL without the token and leaves gzipped events alone", async () => {
    const posthog = await loadPage("/reset-password?token=SECRET&next=%2Fhome");
    const pageURL = window.location.href;
    const gzipped = { type: 2, cv: "2024-10", data: `\u001f\u008b?token=x` };

    posthog.capture(
      "$snapshot",
      {
        $snapshot_data: [
          { type: 4, data: { href: pageURL, width: 1, height: 1 } },
          {
            type: 5,
            data: { tag: "$url_changed", payload: { href: pageURL } },
          },
          gzipped,
        ],
      },
      { _batchKey: "recordings" },
    );
    const snapshot = await waitForEvent("$snapshot");

    const [meta, urlChanged, compressed] = snapshot.properties.$snapshot_data;
    expect(meta.data.href).toMatch(/\?token=\[REDACTED\]&next=%2Fhome$/);
    expect(urlChanged.data.payload.href).toBe(meta.data.href);
    expect(compressed).toEqual(gzipped);
    expectNoSecret(snapshot);
  });
});

const USER = { id: "user-1", email: "ada@example.com" } as unknown as User;
const SIGNED_LINK_REDIRECT =
  "/login?next=%2Fsettings%2Faccount%3Ff%3Doff%26t%3DSECRET";
const MASKED_NEXT = "next=%2Fsettings%2Faccount%3Ff%3Doff%26t%3D%5BREDACTED%5D";
const LINKING_PAGE = `${PAGE_ORIGIN}/link/SECRET`;

async function loadPage(
  path: string,
  { user = null as User | null, trackers = true } = {},
) {
  const url = new URL(withSecret(path), PAGE_ORIGIN);
  setPageURL(url.href);
  // A client from an earlier test can still write its storage, keyed by token.
  pageLoads += 1;
  vi.stubEnv("NEXT_PUBLIC_POSTHOG_KEY", `phc_test_${pageLoads}`);
  vi.resetModules();
  const client = new PostHog();
  clients.push(client);
  vi.doMock("posthog-js", () => ({
    default: client,
    posthog: client,
    PostHog,
  }));
  vi.doMock("@/lib/auth/hooks/useAuth", () => ({
    useAuth: () => ({ user, isUserLoading: false }),
  }));
  vi.doMock("next/navigation", () => ({
    usePathname: () => url.pathname,
    useSearchParams: () => url.searchParams,
  }));
  const [provider, { setAnalyticsIdentitySignedIn }, captureQueue] =
    await Promise.all([
      import("@/providers/posthog/posthog-provider"),
      import("@/services/analytics/anonymous-id"),
      import("@/services/analytics/posthog-capture"),
    ]);
  captureQueueResets.push(captureQueue.resetPostHogCaptureQueueForTests);
  setAnalyticsIdentitySignedIn(user?.id ?? null);
  // Before render: a signed-in visitor is identified during it.
  client.on("eventCaptured", (event) => captured.push(event));
  const { PostHogProvider, PostHogPageViewTracker, PostHogUserTracker } =
    provider;
  render(
    <PostHogProvider>
      {trackers && (
        <>
          <PostHogUserTracker />
          <PostHogPageViewTracker />
        </>
      )}
    </PostHogProvider>,
  );
  return client;
}

// What an earlier visit, or agpt.co, left in posthog-js's storage, written by
// posthog-js itself with nothing masked.
function storeAsEarlierVisit(path: string, referrer?: string) {
  setPageURL(new URL(withSecret(path), PAGE_ORIGIN).href);
  setReferrer(referrer);
  const earlier = new PostHog();
  earlier.init(`phc_test_${pageLoads + 1}`, {
    api_host: POSTHOG_ORIGIN,
    persistence: "localStorage+cookie",
    advanced_disable_flags: true,
    capture_pageview: false,
    autocapture: false,
    disable_session_recording: true,
  });
  earlier.capture("probe");
  const storage = storedItems();
  const cookies = document.cookie.split("; ").filter(Boolean);
  // Retire it without letting it write again; its memory switch clears both.
  earlier.set_config({ persistence: "memory" });
  Object.entries(storage).forEach(([name, value]) =>
    window.localStorage.setItem(name, value),
  );
  cookies.forEach((cookie) => {
    document.cookie = `${cookie}; Path=/`;
  });
  setReferrer(undefined);
}

async function waitForStoredPageInFlags() {
  const key = `phc_test_${pageLoads}`;
  await vi.waitFor(() =>
    expect(
      flagsBodies.some(
        (body) =>
          body.token === key && body.person_properties?.$initial_current_url,
      ),
    ).toBe(true),
  );
  const ownBodies = flagsBodies.filter((body) => body.token === key);
  expect(JSON.stringify(ownBodies)).not.toContain(LINK_VALUE);
  return ownBodies.find((body) => body.person_properties?.$initial_current_url)!
    .person_properties!;
}

function expectPostHogStorageHoldsNoSecret() {
  const postHogItems = Object.entries(storedItems()).filter(([name]) =>
    name.startsWith("ph_"),
  );
  expect(JSON.stringify(postHogItems)).not.toContain(LINK_VALUE);
  expect(document.cookie).not.toContain(LINK_VALUE);
}

async function waitForEvent(name: string, timeout?: number) {
  await vi.waitFor(
    () => expect(captured.map((event) => event.event)).toContain(name),
    { timeout },
  );
  return captured.find((event) => event.event === name)!;
}

function expectNoSecret(event: CaptureResult) {
  expect(JSON.stringify(event)).not.toContain(LINK_VALUE);
}

interface FlagsBody {
  token?: string;
  person_properties?: Record<string, string>;
}

async function readFlagsBody(request: Request): Promise<FlagsBody> {
  const body = await request.text();
  const data = new URLSearchParams(body).get("data");
  const json = data ? Buffer.from(data, "base64").toString("utf8") : body;
  return JSON.parse(json) as FlagsBody;
}

function storedItems(): Record<string, string> {
  return Object.fromEntries(
    Array.from({ length: window.localStorage.length }, (_, index) => {
      const name = window.localStorage.key(index)!;
      return [name, window.localStorage.getItem(name) ?? ""];
    }),
  );
}

function setReferrer(referrer: string | undefined) {
  if (referrer === undefined) {
    Reflect.deleteProperty(document, "referrer");
    return;
  }
  Object.defineProperty(document, "referrer", {
    configurable: true,
    value: withSecret(referrer),
  });
}

function withSecret(text: string) {
  return text.replaceAll("SECRET", LINK_VALUE);
}

function setPageURL(url: string) {
  const { happyDOM } = window as unknown as {
    happyDOM?: { setURL: (url: string) => void };
  };
  happyDOM?.setURL(url);
}
