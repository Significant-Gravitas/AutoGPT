import {
  afterAll,
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";

// Read once at load by `flag-backend`, so it has to be set before the imports below.
const previousBackend = vi.hoisted(() => {
  const previous = process.env.NEXT_PUBLIC_FEATURE_FLAG_BACKEND;
  process.env.NEXT_PUBLIC_FEATURE_FLAG_BACKEND = "server";
  return previous;
});

import { getGetFeatureFlagsMockHandler } from "@/app/api/__generated__/endpoints/feature-flags/feature-flags.msw";
import type { FrontendFlagValues } from "@/app/api/__generated__/models/frontendFlagValues";
import openapi from "@/app/api/openapi.json";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { server } from "@/mocks/mock-server";
import { environment } from "@/services/environment";
import {
  answerCookiebot,
  configureCookiebot,
  installCookiebot,
  removeCookiebot,
} from "@/tests/integrations/cookiebot";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import type { ReactNode } from "react";
import { Flag, useFlagStatus } from "../use-get-flag";

vi.mock("@/app/(platform)/marketplace/components/HeroSection/helpers", () => ({
  DEFAULT_SEARCH_TERMS: [],
}));

const HIRE_EXPERTS = "hire-experts";
const VISITOR = "0b6c1f3e-visitor-0001";

vi.mock("posthog-js", () => ({
  default: { __loaded: true, get_distinct_id: () => VISITOR },
}));

beforeEach(() => {
  signIn("u-1");
});

afterAll(() => {
  if (previousBackend === undefined) {
    delete process.env.NEXT_PUBLIC_FEATURE_FLAG_BACKEND;
  } else {
    process.env.NEXT_PUBLIC_FEATURE_FLAG_BACKEND = previousBackend;
  }
  useAuthStore.setState({ user: null, isUserLoading: true });
});

describe("server backend", () => {
  it("serves the server's answer, and a flag it left out is not answered yet", async () => {
    serveFlags({ flags: { [HIRE_EXPERTS]: true }, configured: true });
    const { result } = renderHook(
      () => ({
        hire: useFlagStatus(Flag.HIRE_EXPERTS),
        search: useFlagStatus(Flag.CHAT_SEARCH),
      }),
      { wrapper: withQueryClient() },
    );

    await waitFor(() => expect(result.current.hire.answered).toBe(true));
    expect(result.current.hire.enabled).toBe(true);
    // A gated route must keep waiting rather than 404 on a flag the vendor never answered.
    expect(result.current.search).toMatchObject({
      enabled: false,
      answered: false,
    });
  });

  it("answers every flag with its default when the server has no vendor", async () => {
    serveFlags({ flags: {}, configured: false });

    const { result } = renderHook(() => useFlagStatus(Flag.CHAT_SEARCH), {
      wrapper: withQueryClient(),
    });

    await waitFor(() => expect(result.current.answered).toBe(true));
    expect(result.current.enabled).toBe(false);
  });

  describe("while the server has no answer for a flag", () => {
    beforeEach(() => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it("asks again until the vendor answers it, then stops", async () => {
      const answers = [{}, {}, { [HIRE_EXPERTS]: true }];
      let requests = 0;
      serveFlags(() => ({
        flags: answers[Math.min(requests++, answers.length - 1)],
        configured: true,
      }));

      const { result } = renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS), {
        wrapper: withQueryClient(),
      });
      await waitFor(() => expect(requests).toBe(1));
      await act(() => vi.advanceTimersByTimeAsync(60_000));

      expect(result.current).toMatchObject({ enabled: true, answered: true });
      const answeredAfter = requests;
      await act(() => vi.advanceTimersByTimeAsync(10 * 60_000));
      expect(requests).toBe(answeredAfter);
    });

    it("serves the default at once for a flag the vendor has no definition of", async () => {
      let requests = 0;
      serveFlags(() => {
        requests++;
        return { flags: { [HIRE_EXPERTS]: null }, configured: true };
      });

      const { result } = renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS), {
        wrapper: withQueryClient(),
      });
      await waitFor(() =>
        expect(result.current).toMatchObject({
          enabled: false,
          answered: true,
        }),
      );
      await act(() => vi.advanceTimersByTimeAsync(10 * 60_000));

      expect(requests).toBe(1);
    });

    it("asks again after a request that failed, until one succeeds", async () => {
      let requests = 0;
      server.use(
        http.get("/api/proxy/api/feature-flags", () =>
          requests++ === 0
            ? HttpResponse.json({ detail: "bad gateway" }, { status: 502 })
            : HttpResponse.json({
                flags: { [HIRE_EXPERTS]: true },
                configured: true,
              }),
        ),
      );

      const { result } = renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS), {
        wrapper: withQueryClient(),
      });
      await waitFor(() => expect(requests).toBe(1));
      await act(() => vi.advanceTimersByTimeAsync(60_000));

      expect(result.current).toMatchObject({ enabled: true, answered: true });
    });

    it("backs off on failed requests too, so an outage is not hammered", async () => {
      let requests = 0;
      server.use(
        http.get("/api/proxy/api/feature-flags", () => {
          requests++;
          return HttpResponse.json({ detail: "bad gateway" }, { status: 502 });
        }),
      );

      renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS), {
        wrapper: withQueryClient(),
      });
      await waitFor(() => expect(requests).toBe(1));
      await act(() => vi.advanceTimersByTimeAsync(10 * 60_000));

      // Every 2.5 s would be 240; backing off to a minute is a dozen or so.
      expect(requests).toBeGreaterThan(4);
      expect(requests).toBeLessThan(20);
    });

    it("backs off to about once a minute when it never answers", async () => {
      let requests = 0;
      serveFlags(() => {
        requests++;
        return { flags: {}, configured: true };
      });

      renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS), {
        wrapper: withQueryClient(),
      });
      await waitFor(() => expect(requests).toBe(1));
      await act(() => vi.advanceTimersByTimeAsync(10 * 60_000));

      // Every 5 s would be 120; backing off to a minute is a dozen or so.
      expect(requests).toBeGreaterThan(4);
      expect(requests).toBeLessThan(20);
    });

    it.each([
      [
        "a flag only another component reads",
        { flags: { [HIRE_EXPERTS]: true }, configured: true },
      ],
      [
        "any flag, with no vendor to wait for",
        { flags: {}, configured: false },
      ],
    ])("never asks again over %s", async (_, response) => {
      let requests = 0;
      serveFlags(() => {
        requests++;
        return response;
      });

      renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS), {
        wrapper: withQueryClient(),
      });
      await waitFor(() => expect(requests).toBe(1));
      await act(() => vi.advanceTimersByTimeAsync(10 * 60_000));

      expect(requests).toBe(1);
    });
  });

  it("never shows one user's flags to the next while theirs load", async () => {
    let requests = 0;
    server.use(
      getGetFeatureFlagsMockHandler(async () => {
        if (requests++ === 0) {
          return { flags: { [HIRE_EXPERTS]: true }, configured: true };
        }
        await new Promise((resolve) => setTimeout(resolve, 200));
        return { flags: { [HIRE_EXPERTS]: false }, configured: true };
      }),
    );
    const seen = renderRecording(() => useFlagStatus(Flag.HIRE_EXPERTS));
    await waitFor(() =>
      expect(seen.at(-1)).toMatchObject({ enabled: true, answered: true }),
    );
    const before = seen.length;

    act(() => signIn("u-2"));
    await waitFor(() =>
      expect(seen.at(-1)).toMatchObject({ enabled: false, answered: true }),
    );

    expect(seen.slice(before).some((s) => s.enabled && s.answered)).toBe(false);
  });

  it("waits for the session, then reads each signed-in user's own flags", async () => {
    const answers = [true, false];
    let requests = 0;
    serveFlags(() => ({
      flags: { [HIRE_EXPERTS]: answers[requests++] },
      configured: true,
    }));
    useAuthStore.setState({ isUserLoading: true });

    const { result } = renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS), {
      wrapper: withQueryClient(),
    });
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(requests).toBe(0);

    act(() => signIn("u-1"));
    await waitFor(() => expect(requests).toBe(1));
    await waitFor(() => expect(result.current.enabled).toBe(true));
    act(() => signIn("u-2"));
    await waitFor(() => expect(requests).toBe(2));
    await waitFor(() =>
      expect(result.current).toMatchObject({ enabled: false, answered: true }),
    );
  });
});

describe("a signed-out visitor", () => {
  beforeEach(() => {
    useAuthStore.setState({ user: null, isUserLoading: false });
    vi.spyOn(environment, "isPostHogEnabled").mockReturnValue("phc_test");
    configureCookiebot();
  });

  afterEach(() => {
    removeCookiebot();
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  it("who allowed analytics is read under their own PostHog id", async () => {
    installCookiebot({ statistics: true });
    const sent = recordVisitorIDs();

    renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS), {
      wrapper: withQueryClient(),
    });

    await waitFor(() => expect(sent).toEqual([VISITOR]));
  });

  it.each([
    ["declined", { statistics: false }],
    ["has not answered", undefined],
  ])("who %s sends no id", async (_, answer) => {
    installCookiebot(answer);
    const sent = recordVisitorIDs();

    renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS), {
      wrapper: withQueryClient(),
    });

    await waitFor(() => expect(sent).toEqual([null]));
  });

  it("sends no id once signed in, consent or not", async () => {
    installCookiebot({ statistics: true });
    signIn("u-1");
    const sent = recordVisitorIDs();

    renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS), {
      wrapper: withQueryClient(),
    });

    await waitFor(() => expect(sent).toEqual([null]));
  });

  it("keeps the shared answers until their own land when they allow analytics", async () => {
    installCookiebot();
    server.use(
      getGetFeatureFlagsMockHandler(async ({ request }) => {
        if (!new URL(request.url).searchParams.get("visitor_id")) {
          return { flags: { [HIRE_EXPERTS]: false }, configured: true };
        }
        await new Promise((resolve) => setTimeout(resolve, 200));
        return { flags: { [HIRE_EXPERTS]: true }, configured: true };
      }),
    );
    const seen = renderRecording(() => useFlagStatus(Flag.HIRE_EXPERTS));
    await waitFor(() =>
      expect(seen.at(-1)).toMatchObject({ enabled: false, answered: true }),
    );
    const before = seen.length;

    act(() => answerCookiebot({ statistics: true }));
    await waitFor(() => expect(seen.at(-1)).toMatchObject({ enabled: true }));

    // A gate must not drop back to "not answered yet" while their own load.
    expect(seen.slice(before).every((s) => s.answered)).toBe(true);
  });

  it("is read again under their id when they allow analytics", async () => {
    installCookiebot();
    const sent = recordVisitorIDs();
    renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS), {
      wrapper: withQueryClient(),
    });
    await waitFor(() => expect(sent).toEqual([null]));

    act(() => answerCookiebot({ statistics: true }));

    await waitFor(() => expect(sent).toEqual([null, VISITOR]));
  });
});

describe("the frontend's flags are the set the endpoint serves", () => {
  it("keys the endpoint's flags by the same enum the frontend reads from", () => {
    const schemas = openapi.components.schemas;
    const response =
      openapi.paths["/api/feature-flags"].get.responses["200"].content[
        "application/json"
      ].schema.$ref;

    expect(response).toBe("#/components/schemas/FrontendFlagValues");
    expect(schemas.FrontendFlagValues.properties.flags.propertyNames.$ref).toBe(
      "#/components/schemas/FrontendFlag",
    );
    expect([...schemas.FrontendFlag.enum].sort()).toEqual(
      Object.values(Flag).sort(),
    );
  });
});

function renderRecording<T>(read: () => T) {
  const seen: T[] = [];
  renderHook(
    () => {
      const value = read();
      seen.push(value);
      return value;
    },
    { wrapper: withQueryClient() },
  );
  return seen;
}

function recordVisitorIDs() {
  const sent: Array<string | null> = [];
  server.use(
    getGetFeatureFlagsMockHandler(({ request }) => {
      sent.push(new URL(request.url).searchParams.get("visitor_id"));
      return { flags: {}, configured: true };
    }),
  );
  return sent;
}

function serveFlags(response: FrontendFlagValues | (() => FrontendFlagValues)) {
  server.use(getGetFeatureFlagsMockHandler(response));
}

function signIn(userID: string) {
  useAuthStore.setState({
    user: { id: userID, email: `${userID}@example.com`, user_metadata: {} },
    isUserLoading: false,
  });
}

function withQueryClient() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  };
}
