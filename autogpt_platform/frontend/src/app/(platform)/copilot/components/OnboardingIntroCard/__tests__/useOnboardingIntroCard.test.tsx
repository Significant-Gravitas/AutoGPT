import { useGetBrainDumpSuggestedTasks } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import { getGetBrainDumpIntroMockHandler200 } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump.msw";
import type { IntroCardResponse } from "@/app/api/__generated__/models/introCardResponse";
import { server } from "@/mocks/mock-server";
import { setIntroPath } from "@/services/onboarding/brain-dump-handoff";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { HttpResponse, http } from "msw";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const capture = vi.hoisted(() => vi.fn());
vi.mock("posthog-js", () => ({ default: { capture } }));

const authUser = vi.hoisted(() => ({
  current: { id: "user-1" } as { id: string } | null,
}));
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: authUser.current }),
}));

import { useOnboardingIntroCard } from "../useOnboardingIntroCard";

const INTRO_URL =
  "http://localhost:3000/api/proxy/api/onboarding/brain-dump/intro";
const KICKOFF_URL =
  "http://localhost:3000/api/proxy/api/onboarding/brain-dump/suggested-tasks/kickoff";
const TASKS_URL =
  "http://localhost:3000/api/proxy/api/onboarding/brain-dump/suggested-tasks";
const MIC_GLOW_KEY = "autogpt:onboarding-mic-glow";
const LATER_DUMP_KEY = "autogpt:onboarding-pending-later-dump";
const FOLLOWUP_KEY = "autogpt:onboarding-intro-awaiting-followup";
const WELCOME_PENDING_KEY = "autogpt:onboarding-welcome-pending";
const GREETING_DONE_KEY = "autogpt:copilot-greeting-done";

const READY_INTRO: IntroCardResponse = {
  path: "A",
  greeting: "You spend most of your week chasing updates.",
  greeting_done: false,
  prompts: [{ title: "Summarise my inbox", prompt: "Summarise my inbox" }],
  transcript: "I rebuild the same report every Monday.",
};

const PENDING_INTRO: IntroCardResponse = {
  path: "A",
  greeting: "",
  greeting_done: false,
  greeting_pending: true,
  prompts: [],
};

function makeWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  };
}

function renderIntro() {
  return renderHook(() => useOnboardingIntroCard(), { wrapper: makeWrapper() });
}

// Counts every intro request so "the endpoint is never called" means silence
// rather than a handler that happens not to be registered.
function countIntroRequests(body: IntroCardResponse = READY_INTRO) {
  const urls: string[] = [];
  server.use(
    http.get(INTRO_URL, ({ request }) => {
      urls.push(request.url);
      return HttpResponse.json(body);
    }),
  );
  return urls;
}

beforeEach(() => {
  window.sessionStorage.clear();
  window.localStorage.clear();
  capture.mockReset();
  authUser.current = { id: "user-1" };
});

describe("useOnboardingIntroCard — availability", () => {
  it("holds the composer until the intro answers", async () => {
    // The hold stops the regular hero flashing before the greeting page
    // takes over, so it lasts until the server has said something.
    let answer: () => void = () => {};
    const answered = new Promise<void>((resolve) => {
      answer = resolve;
    });
    server.use(
      http.get(INTRO_URL, async () => {
        await answered;
        return HttpResponse.json(READY_INTRO);
      }),
    );

    const { result } = renderIntro();
    await act(async () => {});
    expect(result.current.isAwaitingGreeting).toBe(true);
    expect(result.current.isVisible).toBe(false);

    answer();
    await waitFor(() => expect(result.current.isVisible).toBe(true));
    expect(result.current.isAwaitingGreeting).toBe(false);
  });

  it("never asks again once this browser has seen this user finish the greeting", async () => {
    window.localStorage.setItem(GREETING_DONE_KEY, "user-1");
    const urls = countIntroRequests();

    const { result } = renderIntro();
    await waitFor(() => expect(result.current.isAwaitingGreeting).toBe(false));

    expect(urls).toEqual([]);
    expect(result.current.isVisible).toBe(false);
  });

  it("still asks when the cached completion belongs to a different account", async () => {
    window.localStorage.setItem(GREETING_DONE_KEY, "someone-else");
    const urls = countIntroRequests();

    const { result } = renderIntro();
    await waitFor(() => expect(result.current.isVisible).toBe(true));

    expect(urls).toHaveLength(1);
  });
});

describe("useOnboardingIntroCard — handoff from the wizard", () => {
  it("consumes the recorded path once and reports it", async () => {
    setIntroPath("A");
    countIntroRequests();

    const { result } = renderIntro();
    await waitFor(() => expect(result.current.isVisible).toBe(true));

    expect(window.sessionStorage.getItem("autogpt:onboarding-intro-path")).toBe(
      null,
    );
    expect(window.sessionStorage.getItem(FOLLOWUP_KEY)).toBe("1");
    expect(capture).toHaveBeenCalledWith("intro_path", { path: "A" });
    // Path A already dumped — nothing to point the mic at.
    expect(window.sessionStorage.getItem(MIC_GLOW_KEY)).toBeNull();
  });

  it("glows the composer mic only for the skip path, and arms the later dump", async () => {
    setIntroPath("B");
    countIntroRequests();

    const { result } = renderIntro();
    await waitFor(() => expect(result.current.isVisible).toBe(true));

    expect(window.sessionStorage.getItem(MIC_GLOW_KEY)).toBe("1");
    // The invitation Otto just issued: the first voice message in the
    // composer is the dump this user skipped, and is reported as such.
    expect(window.sessionStorage.getItem(LATER_DUMP_KEY)).toBe("1");
    expect(capture).toHaveBeenCalledWith("intro_path", { path: "B" });
  });

  it("reports no path for a user who did not just come out of onboarding", async () => {
    countIntroRequests();

    const { result } = renderIntro();
    await waitFor(() => expect(result.current.isVisible).toBe(true));

    expect(capture).not.toHaveBeenCalledWith("intro_path", expect.anything());
  });
});

describe("useOnboardingIntroCard — first landing", () => {
  it("fetches the greeting as soon as it mounts", async () => {
    setIntroPath("A");
    const urls = countIntroRequests();

    const { result } = renderIntro();
    await waitFor(() => expect(result.current.isVisible).toBe(true));

    expect(result.current.greeting).toBe(READY_INTRO.greeting);
    expect(result.current.isAwaitingGreeting).toBe(false);
    expect(urls).toHaveLength(1);
  });

  it("ignores the welcome flag an earlier version left in the session", async () => {
    window.sessionStorage.setItem(WELCOME_PENDING_KEY, "1");
    const urls = countIntroRequests();

    const { result } = renderIntro();
    await waitFor(() => expect(result.current.isVisible).toBe(true));

    expect(urls).toHaveLength(1);
  });
});

describe("useOnboardingIntroCard — the greeting itself", () => {
  it("exposes the server's greeting, transcript and path", async () => {
    server.use(getGetBrainDumpIntroMockHandler200(READY_INTRO));

    const { result } = renderIntro();
    await waitFor(() => expect(result.current.isVisible).toBe(true));

    expect(result.current.greeting).toBe(READY_INTRO.greeting);
    expect(result.current.transcript).toBe(READY_INTRO.transcript);
    expect(result.current.path).toBe("A");
    expect(result.current.isAwaitingGreeting).toBe(false);
  });

  it("treats a non-string transcript as no transcript", async () => {
    server.use(
      getGetBrainDumpIntroMockHandler200({ ...READY_INTRO, transcript: null }),
    );

    const { result } = renderIntro();
    await waitFor(() => expect(result.current.isVisible).toBe(true));

    expect(result.current.transcript).toBe("");
  });

  it("defaults the path to B when the payload does not carry one", async () => {
    server.use(
      getGetBrainDumpIntroMockHandler200({
        greeting: "Tell me what eats your week.",
        prompts: [],
      } as unknown as IntroCardResponse),
    );

    const { result } = renderIntro();
    await waitFor(() => expect(result.current.isVisible).toBe(true));

    expect(result.current.path).toBe("B");
  });

  it("holds the composer back while the pipeline is still writing the greeting", async () => {
    server.use(getGetBrainDumpIntroMockHandler200(PENDING_INTRO));

    const { result } = renderIntro();
    await waitFor(() => expect(result.current.isAwaitingGreeting).toBe(true));

    expect(result.current.isVisible).toBe(false);
    expect(result.current.greeting).toBe("");
    // The orb sits centered for exactly this window; only the greeting
    // itself anchors the page to the top.
    expect(result.current.anchorTop).toBe(false);
  });

  it("releases the loader when the server sends no greeting and no pending flag", async () => {
    // A terminally-empty Path A (generation gave up server-side): there is
    // nothing coming, so holding the orb up would strand the composer.
    server.use(
      getGetBrainDumpIntroMockHandler200({
        path: "A",
        greeting: "",
        greeting_done: false,
        greeting_pending: false,
        prompts: [],
      }),
    );

    const { result } = renderIntro();
    await waitFor(() => expect(result.current.isAwaitingGreeting).toBe(false));

    expect(result.current.isVisible).toBe(false);
  });

  it("polls past the pending answer and reveals the greeting when it lands", async () => {
    let calls = 0;
    server.use(
      http.get(INTRO_URL, () => {
        calls += 1;
        return HttpResponse.json(calls === 1 ? PENDING_INTRO : READY_INTRO);
      }),
    );

    const { result } = renderIntro();
    await waitFor(() => expect(result.current.isAwaitingGreeting).toBe(true));

    await waitFor(() => expect(result.current.isVisible).toBe(true), {
      timeout: 6000,
    });
    expect(calls).toBeGreaterThan(1);
    expect(result.current.greeting).toBe(READY_INTRO.greeting);
  }, 10_000);

  it("shows nothing and caches the server's 'already seen it' verdict", async () => {
    const urls = countIntroRequests({
      path: "A",
      greeting: "Welcome back.",
      greeting_done: true,
    });

    const { result } = renderIntro();
    await waitFor(() =>
      expect(window.localStorage.getItem(GREETING_DONE_KEY)).toBe("user-1"),
    );

    expect(result.current.isVisible).toBe(false);
    expect(result.current.isAwaitingGreeting).toBe(false);
    expect(urls).toHaveLength(1);
  });

  it("never paints the card for a done verdict that still carries a greeting", async () => {
    // The verdict is cached in an effect, which runs after the render that
    // first saw it — reading only the cached copy flashed the card for the
    // paint in between.
    countIntroRequests({
      path: "A",
      greeting: "Welcome back.",
      greeting_done: true,
    });
    const painted: boolean[] = [];

    renderHook(
      () => {
        const value = useOnboardingIntroCard();
        painted.push(value.isVisible);
        return value;
      },
      { wrapper: makeWrapper() },
    );
    await waitFor(() =>
      expect(window.localStorage.getItem(GREETING_DONE_KEY)).toBe("user-1"),
    );

    expect(painted).not.toContain(true);
  });

  it("writes no cache entry while the user record is still loading", async () => {
    authUser.current = null;
    countIntroRequests({
      path: "A",
      greeting: "Welcome back.",
      greeting_done: true,
    });

    const { result } = renderIntro();
    await waitFor(() => expect(result.current.isAwaitingGreeting).toBe(false));

    // A cache entry keyed to nobody would be inherited by whoever signs in.
    expect(window.localStorage.getItem(GREETING_DONE_KEY)).toBeNull();
  });

  it("releases the composer when the endpoint is unavailable", async () => {
    server.use(
      http.get(INTRO_URL, () =>
        HttpResponse.json({ detail: "nope" }, { status: 500 }),
      ),
    );

    const { result } = renderIntro();
    await waitFor(() => expect(result.current.isAwaitingGreeting).toBe(false));

    expect(result.current.isVisible).toBe(false);
    expect(result.current.greeting).toBe("");
  });
});

describe("useOnboardingIntroCard — team selection belongs to the wizard", () => {
  it.each([false, true])(
    "does not poll for a team after the greeting settles (done=%s)",
    async (greetingDone) => {
      const requests = countIntroRequests({
        ...READY_INTRO,
        greeting_done: greetingDone,
        team_pending: true,
      });
      const { result } = renderIntro();
      await waitFor(() => expect(requests).toHaveLength(1));
      await waitFor(() =>
        expect(result.current.isAwaitingGreeting).toBe(false),
      );
      expect(result.current.isVisible).toBe(!greetingDone);
      await new Promise((resolve) => setTimeout(resolve, 3200));
      expect(requests).toHaveLength(1);
    },
    10_000,
  );
});

describe("useOnboardingIntroCard — starting the suggested tasks", () => {
  const GREETING_POLL_MS = 1_500;

  afterEach(() => {
    vi.useRealTimers();
  });

  async function advance(ms: number) {
    await act(async () => {
      await vi.advanceTimersByTimeAsync(ms);
    });
  }

  // Counts kickoff calls and suggested-task reads, the latter through a
  // list mounted beside the hook as the home page mounts it.
  function trackKickoff(status: "started" | "pending" | "nothing") {
    const calls = { kickoffs: 0, taskReads: 0 };
    server.use(
      http.post(KICKOFF_URL, () => {
        calls.kickoffs += 1;
        return HttpResponse.json({ status, started_task_ids: [] });
      }),
      http.get(TASKS_URL, () => {
        calls.taskReads += 1;
        return HttpResponse.json({ tasks: [], stored: true });
      }),
    );
    return calls;
  }

  function renderWithTasks() {
    return renderHook(
      () => {
        useGetBrainDumpSuggestedTasks();
        return useOnboardingIntroCard();
      },
      { wrapper: makeWrapper() },
    );
  }

  it("retries kickoff when the greeting arrives", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    let introCalls = 0;
    server.use(
      http.get(INTRO_URL, () => {
        introCalls += 1;
        return HttpResponse.json(
          introCalls === 1 ? PENDING_INTRO : READY_INTRO,
        );
      }),
    );
    const calls = trackKickoff("nothing");

    const { result } = renderWithTasks();
    await waitFor(() => expect(result.current.isAwaitingGreeting).toBe(true));
    await waitFor(() => expect(calls.taskReads).toBe(1));
    expect(calls.kickoffs).toBe(0);

    await advance(GREETING_POLL_MS);

    await waitFor(() => expect(result.current.isVisible).toBe(true));
    await waitFor(() => expect(calls.kickoffs).toBe(1));
    // Settled with "nothing", yet a background kickoff may have started the
    // tasks meanwhile, so the list is read again.
    await waitFor(() => expect(calls.taskReads).toBe(2));

    await advance(10 * GREETING_POLL_MS);
    expect(calls.kickoffs).toBe(1);
  });

  it("does not kick off for a user who already saw the greeting", async () => {
    countIntroRequests({
      path: "A",
      greeting: "Welcome back.",
      greeting_done: true,
    });
    const calls = trackKickoff("started");

    const { result } = renderWithTasks();
    await waitFor(() =>
      expect(window.localStorage.getItem(GREETING_DONE_KEY)).toBe("user-1"),
    );
    await act(async () => {});

    expect(result.current.isVisible).toBe(false);
    expect(calls.kickoffs).toBe(0);
  });
});
