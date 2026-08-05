import { getGetBrainDumpIntroMockHandler200 } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump.msw";
import type { IntroCardResponse } from "@/app/api/__generated__/models/introCardResponse";
import { server } from "@/mocks/mock-server";
import { setIntroPath } from "@/services/onboarding/brain-dump-handoff";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { HttpResponse, http } from "msw";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const capture = vi.hoisted(() => vi.fn());
vi.mock("posthog-js", () => ({ default: { capture } }));

const authUser = vi.hoisted(() => ({
  current: { id: "user-1" } as { id: string } | null,
}));
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: authUser.current }),
}));

const flags = vi.hoisted(() => ({ current: {} as Record<string, boolean> }));
const flagsReady = vi.hoisted(() => ({ current: true }));
vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: { ONBOARDING_BRAIN_DUMP: "onboarding-brain-dump" },
  useGetFlag: (flag: string) => flags.current[flag] ?? false,
  useFlagStatus: (flag: string) => ({
    enabled: flags.current[flag] ?? false,
    ready: flagsReady.current,
  }),
}));

import { useOnboardingIntroCard } from "../useOnboardingIntroCard";

const INTRO_URL =
  "http://localhost:3000/api/proxy/api/onboarding/brain-dump/intro";
const MIC_GLOW_KEY = "autogpt:onboarding-mic-glow";
const LATER_DUMP_KEY = "autogpt:onboarding-pending-later-dump";
const FOLLOWUP_KEY = "autogpt:onboarding-intro-awaiting-followup";
const WELCOME_PENDING_KEY = "autogpt:onboarding-welcome-pending";
const GREETING_DONE_KEY = "autogpt:copilot-greeting-done";
const CAPABILITY_CARDS_KEY = "autogpt:copilot-capability-cards-seen";

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
  flags.current = { "onboarding-brain-dump": true };
  flagsReady.current = true;
});

describe("useOnboardingIntroCard — flag gating", () => {
  it("never asks for an intro while the brain-dump flag is off", async () => {
    flags.current = {};
    const urls = countIntroRequests();

    const { result } = renderIntro();
    await waitFor(() => expect(result.current.isVisible).toBe(false));

    expect(urls).toEqual([]);
    expect(result.current.greeting).toBe("");
  });

  it("releases the composer once LaunchDarkly says the flag is off", async () => {
    // The regression: with the flag off the intro query is disabled, so it
    // never answers — and "no answer yet" was read as "still generating".
    // isAwaitingGreeting stayed true forever and EmptySession hides the
    // composer, PulseChips and SuggestionThemes behind it, so the default
    // flag-off copilot rendered a hero with no way to type.
    flags.current = {};

    const { result } = renderIntro();

    await waitFor(() => expect(result.current.isAwaitingGreeting).toBe(false));
    expect(result.current.isVisible).toBe(false);
  });

  it("renders the plain hero while LaunchDarkly has not answered for a user with no handoff", async () => {
    // Holding on "no answer yet" alone hid the composer on every flag-off
    // /copilot load until LaunchDarkly replied — the flag-off page has to
    // render untouched.
    flags.current = {};
    flagsReady.current = false;

    const { result } = renderIntro();

    await waitFor(() => expect(result.current.isAwaitingGreeting).toBe(false));
    expect(result.current.isVisible).toBe(false);
  });

  it("still holds the composer while LaunchDarkly has not answered for a user coming out of the wizard", async () => {
    // The hold exists to stop the regular hero flashing before the
    // greeting page takes over, and the pending overlay is seeded
    // synchronously — so this user is held before the flag resolves.
    window.sessionStorage.setItem(WELCOME_PENDING_KEY, "1");
    flags.current = {};
    flagsReady.current = false;

    const { result } = renderIntro();

    await waitFor(() => expect(result.current).toBeDefined());
    expect(result.current.isAwaitingGreeting).toBe(true);
  });

  it("never asks again once this browser has seen this user finish the greeting", async () => {
    window.localStorage.setItem(GREETING_DONE_KEY, "user-1");
    const urls = countIntroRequests();

    const { result } = renderIntro();
    await waitFor(() => expect(result.current.isAwaitingGreeting).toBe(false));

    expect(urls).toEqual([]);
    expect(result.current.isVisible).toBe(false);
  });

  it("still asks when the cached flag belongs to a different account", async () => {
    window.localStorage.setItem(GREETING_DONE_KEY, "someone-else");
    const urls = countIntroRequests();

    const { result } = renderIntro();
    await waitFor(() => expect(result.current.isVisible).toBe(true));

    expect(urls).toHaveLength(1);
  });
});

describe("useOnboardingIntroCard — handoff from the wizard", () => {
  it("consumes the recorded path once, opens the welcome overlay and reports it", async () => {
    setIntroPath("A");
    countIntroRequests();

    const { result } = renderIntro();
    await waitFor(() => expect(result.current.isWelcomeOpen).toBe(true));

    expect(window.sessionStorage.getItem("autogpt:onboarding-intro-path")).toBe(
      null,
    );
    expect(window.sessionStorage.getItem(WELCOME_PENDING_KEY)).toBe("1");
    expect(window.sessionStorage.getItem(FOLLOWUP_KEY)).toBe("1");
    expect(capture).toHaveBeenCalledWith("intro_path", { path: "A" });
    // Path A already dumped — nothing to point the mic at.
    expect(window.sessionStorage.getItem(MIC_GLOW_KEY)).toBeNull();
  });

  it("glows the composer mic only for the skip path, and arms the later dump", async () => {
    setIntroPath("B");
    countIntroRequests();

    const { result } = renderIntro();
    await waitFor(() => expect(result.current.isWelcomeOpen).toBe(true));

    expect(window.sessionStorage.getItem(MIC_GLOW_KEY)).toBe("1");
    // The invitation AutoPilot just issued: the first voice message in the
    // composer is the dump this user skipped, and is reported as such.
    expect(window.sessionStorage.getItem(LATER_DUMP_KEY)).toBe("1");
    expect(capture).toHaveBeenCalledWith("intro_path", { path: "B" });
  });

  it("drops the handoff instead of running it when the flag is off", async () => {
    // The wizard wrote these while the flag was on; a rollback in between
    // must not still buy the user the overlay and its 404-ing polls.
    flags.current = {};
    setIntroPath("B");
    window.sessionStorage.setItem(WELCOME_PENDING_KEY, "1");
    const urls = countIntroRequests();

    const { result } = renderIntro();
    await waitFor(() => expect(result.current.isAwaitingGreeting).toBe(false));

    expect(result.current.isWelcomeOpen).toBe(false);
    expect(window.sessionStorage.getItem(WELCOME_PENDING_KEY)).toBeNull();
    expect(window.sessionStorage.getItem("autogpt:onboarding-intro-path")).toBe(
      null,
    );
    expect(window.sessionStorage.getItem(MIC_GLOW_KEY)).toBeNull();
    expect(window.sessionStorage.getItem(LATER_DUMP_KEY)).toBeNull();
    expect(capture).not.toHaveBeenCalledWith("intro_path", expect.anything());
    expect(urls).toEqual([]);
  });

  it("waits for LaunchDarkly before consuming the handoff", async () => {
    flagsReady.current = false;
    setIntroPath("A");
    countIntroRequests();

    const { result } = renderIntro();
    await waitFor(() => expect(result.current).toBeDefined());

    expect(result.current.isWelcomeOpen).toBe(false);
    expect(window.sessionStorage.getItem("autogpt:onboarding-intro-path")).toBe(
      "A",
    );
    expect(capture).not.toHaveBeenCalledWith("intro_path", expect.anything());
  });

  it("shows no overlay for a user who did not just come out of onboarding", async () => {
    countIntroRequests();

    const { result } = renderIntro();
    await waitFor(() => expect(result.current.isVisible).toBe(true));

    expect(result.current.isWelcomeOpen).toBe(false);
    expect(capture).not.toHaveBeenCalledWith("intro_path", expect.anything());
  });

  it("keeps the overlay up across a refresh via the pending flag", async () => {
    window.sessionStorage.setItem(WELCOME_PENDING_KEY, "1");
    const urls = countIntroRequests();

    const { result } = renderIntro();
    await waitFor(() => expect(result.current.isWelcomeOpen).toBe(true));

    // The greeting is not fetched behind a modal the user has not dismissed.
    expect(urls).toEqual([]);
    expect(result.current.isAwaitingGreeting).toBe(true);
  });

  it("drops a stale pending overlay for a user who already saw the cards", async () => {
    window.sessionStorage.setItem(WELCOME_PENDING_KEY, "1");
    window.localStorage.setItem(CAPABILITY_CARDS_KEY, "user-1");
    countIntroRequests();

    const { result } = renderIntro();
    await waitFor(() => expect(result.current.isWelcomeOpen).toBe(false));

    expect(window.sessionStorage.getItem(WELCOME_PENDING_KEY)).toBeNull();
    await waitFor(() => expect(result.current.isVisible).toBe(true));
  });
});

describe("useOnboardingIntroCard — closing the welcome overlay", () => {
  it("reports the close, remembers the cards, and reveals the greeting", async () => {
    setIntroPath("A");
    const urls = countIntroRequests();

    const { result } = renderIntro();
    await waitFor(() => expect(result.current.isWelcomeOpen).toBe(true));
    // The greeting stays hidden behind the overlay however the fetch went.
    expect(result.current.isVisible).toBe(false);
    expect(result.current.isAwaitingGreeting).toBe(true);

    act(() => result.current.closeWelcome());

    expect(capture).toHaveBeenCalledWith("welcome_dialog_closed", {});
    expect(window.sessionStorage.getItem(WELCOME_PENDING_KEY)).toBeNull();
    expect(window.localStorage.getItem(CAPABILITY_CARDS_KEY)).toBe("user-1");

    await waitFor(() => expect(result.current.isVisible).toBe(true));
    expect(result.current.greeting).toBe(READY_INTRO.greeting);
    expect(urls.length).toBeGreaterThan(0);
  });
});

describe("useOnboardingIntroCard — the greeting itself", () => {
  it("exposes the server's greeting, prompts, transcript and path", async () => {
    server.use(getGetBrainDumpIntroMockHandler200(READY_INTRO));

    const { result } = renderIntro();
    await waitFor(() => expect(result.current.isVisible).toBe(true));

    expect(result.current.greeting).toBe(READY_INTRO.greeting);
    expect(result.current.prompts).toEqual(READY_INTRO.prompts);
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
    expect(result.current.prompts).toEqual([]);
  });

  it("holds the composer back while the pipeline is still writing the greeting", async () => {
    server.use(getGetBrainDumpIntroMockHandler200(PENDING_INTRO));

    const { result } = renderIntro();
    await waitFor(() => expect(result.current.isAwaitingGreeting).toBe(true));

    expect(result.current.isVisible).toBe(false);
    expect(result.current.greeting).toBe("");
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
