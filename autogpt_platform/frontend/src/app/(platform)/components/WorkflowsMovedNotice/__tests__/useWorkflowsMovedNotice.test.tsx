import {
  getGetV1OnboardingStateMockHandler,
  getGetV1OnboardingStateResponseMock,
  getPostV1CompleteOnboardingStepMockHandler,
} from "@/app/api/__generated__/endpoints/onboarding/onboarding.msw";
import type { OnboardingStep } from "@/app/api/__generated__/models/onboardingStep";
import { server } from "@/mocks/mock-server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  act,
  fireEvent,
  render,
  renderHook,
  screen,
  waitFor,
} from "@testing-library/react";
import * as Dialog from "@radix-ui/react-dialog";
import { HttpResponse, http } from "msw";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const context = vi.hoisted(() => ({
  user: { id: "user-1", created_at: "2026-09-01T00:00:00Z" } as {
    id: string;
    created_at?: string;
  } | null,
  pathname: "/copilot",
  search: "",
  ready: true,
  flagReadiness: {} as Record<string, boolean>,
  flags: { "hire-experts": true, "autogpt-new-layout": true } as Record<
    string,
    boolean
  >,
}));
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: context.user }),
}));
vi.mock("next/navigation", () => ({
  usePathname: () => context.pathname,
  useSearchParams: () => new URLSearchParams(context.search),
}));
vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {
    HIRE_EXPERTS: "hire-experts",
    AUTOGPT_NEW_LAYOUT: "autogpt-new-layout",
    ONBOARDING_BRAIN_DUMP: "onboarding-brain-dump",
  },
  useFlagStatus: (flag: string) => ({
    enabled: context.flags[flag],
    ready: context.flagReadiness[flag] ?? context.ready,
  }),
}));

import { useWorkflowsMovedNotice } from "../useWorkflowsMovedNotice";
import { WorkflowsMovedNotice } from "../WorkflowsMovedNotice";
import { useOnboardingIntroCard } from "../../../copilot/components/OnboardingIntroCard/useOnboardingIntroCard";
import {
  peekIntroPath,
  setIntroPath,
} from "@/services/onboarding/brain-dump-handoff";

function mockOnboarding(
  completedSteps: OnboardingStep[] = ["ONBOARDING_COMPLETE"],
) {
  server.use(
    getGetV1OnboardingStateMockHandler(
      getGetV1OnboardingStateResponseMock({ completedSteps }),
    ),
  );
}

function renderGate(
  canShow = true,
  queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  }),
) {
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  }
  return {
    ...renderHook(() => useWorkflowsMovedNotice(canShow), { wrapper: Wrapper }),
    queryClient,
  };
}

function expectDeferred(canShow = true) {
  const { result, queryClient } = renderGate(canShow);
  expect(result.current.isOpen).toBe(false);
  expect(queryClient.isFetching()).toBe(0);
}

beforeEach(() => {
  vi.restoreAllMocks();
  window.localStorage.clear();
  window.sessionStorage.clear();
  context.user = { id: "user-1", created_at: "2026-09-01T00:00:00Z" };
  context.pathname = "/copilot";
  context.search = "";
  context.ready = true;
  context.flagReadiness = {};
  context.flags = { "hire-experts": true, "autogpt-new-layout": true };
  mockOnboarding();
});

describe("migration notice eligibility", () => {
  it("yields to a late tutorial without acknowledging the migration or stealing focus", async () => {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    function Harness({ tutorialOpen }: { tutorialOpen: boolean }) {
      return (
        <QueryClientProvider client={queryClient}>
          <button>Page action</button>
          <Dialog.Root open={tutorialOpen}>
            <Dialog.Portal>
              <Dialog.Content aria-describedby={undefined}>
                <Dialog.Title>First visit tutorial</Dialog.Title>
                <button>Continue tutorial</button>
              </Dialog.Content>
            </Dialog.Portal>
          </Dialog.Root>
          <WorkflowsMovedNotice />
        </QueryClientProvider>
      );
    }
    const { rerender } = render(<Harness tutorialOpen={false} />);
    await screen.findByRole("dialog", { name: "Your agents have moved" });

    rerender(<Harness tutorialOpen />);
    await screen.findByRole("dialog", { name: "First visit tutorial" });
    await waitFor(() =>
      expect(
        document.querySelector("[data-workflows-moved-notice]"),
      ).toBeNull(),
    );
    expect(document.querySelectorAll('[role="dialog"]')).toHaveLength(1);
    expect(document.activeElement).toBe(
      screen.getByRole("button", { name: "Continue tutorial" }),
    );
    expect(
      window.localStorage.getItem("autogpt:workflows-moved:2026-09-21:user-1"),
    ).toBeNull();

    rerender(<Harness tutorialOpen={false} />);
    expect(screen.queryByRole("dialog")).toBeNull();
    context.pathname = "/team";
    rerender(<Harness tutorialOpen={false} />);
    await screen.findByRole("dialog", { name: "Your agents have moved" });
  });

  it.each(["A", "B"] as const)(
    "gives pending welcome path %s priority even when its flag resolves late",
    async (path) => {
      setIntroPath(path);
      context.flagReadiness["onboarding-brain-dump"] = false;
      const queryClient = new QueryClient({
        defaultOptions: { queries: { retry: false } },
      });
      function Wrapper({ children }: { children: ReactNode }) {
        return (
          <QueryClientProvider client={queryClient}>
            {children}
          </QueryClientProvider>
        );
      }
      const { result, rerender } = renderHook(
        () => ({
          welcome: useOnboardingIntroCard(),
          migration: useWorkflowsMovedNotice(),
        }),
        { wrapper: Wrapper },
      );

      await waitFor(() => expect(queryClient.isFetching()).toBe(0));
      await act(async () => {
        await new Promise<void>((resolve) =>
          requestAnimationFrame(() => resolve()),
        );
      });
      expect(result.current.migration.isOpen).toBe(false);
      expect(result.current.welcome.isWelcomeOpen).toBe(false);
      expect(peekIntroPath()).toBe(path);

      context.flags["onboarding-brain-dump"] = true;
      context.flagReadiness["onboarding-brain-dump"] = true;
      rerender();
      await waitFor(() =>
        expect(result.current.welcome.isWelcomeOpen).toBe(true),
      );
      expect(result.current.migration.isOpen).toBe(false);
      expect(peekIntroPath()).toBeNull();

      act(() => result.current.welcome.closeWelcome());
      expect(result.current.welcome.isWelcomeOpen).toBe(false);
      expect(result.current.migration.isOpen).toBe(false);
      context.pathname = "/team";
      rerender();
      await waitFor(() => expect(result.current.migration.isOpen).toBe(true));
    },
  );

  it("defers behind an already open Radix dialog until the next landing visit", async () => {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    function Harness({ notificationOpen }: { notificationOpen: boolean }) {
      return (
        <QueryClientProvider client={queryClient}>
          <Dialog.Root open={notificationOpen}>
            <Dialog.Portal>
              <Dialog.Content aria-describedby={undefined}>
                <Dialog.Title>Stay in the loop</Dialog.Title>
                <button>Not now</button>
              </Dialog.Content>
            </Dialog.Portal>
          </Dialog.Root>
          <WorkflowsMovedNotice />
        </QueryClientProvider>
      );
    }
    const { rerender } = render(<Harness notificationOpen />);
    await screen.findByRole("dialog", { name: "Stay in the loop" });
    await waitFor(() => expect(queryClient.isFetching()).toBe(0));
    await act(async () => {
      await new Promise<void>((resolve) =>
        requestAnimationFrame(() => resolve()),
      );
    });
    expect(
      screen.queryByRole("dialog", { name: "Your agents have moved" }),
    ).toBeNull();
    expect(screen.getAllByRole("dialog")).toHaveLength(1);
    rerender(<Harness notificationOpen={false} />);
    expect(
      screen.queryByRole("dialog", { name: "Your agents have moved" }),
    ).toBeNull();
    context.pathname = "/team";
    rerender(<Harness notificationOpen={false} />);
    await screen.findByRole("dialog", { name: "Your agents have moved" });
    expect(screen.getAllByRole("dialog")).toHaveLength(1);
  });

  it.each(["pointerDown", "keyDown"] as const)(
    "defers late onboarding after %s until the next landing route",
    async (interaction) => {
      let releaseOnboarding = () => {};
      const pending = new Promise<void>((resolve) => {
        releaseOnboarding = resolve;
      });
      server.use(
        getGetV1OnboardingStateMockHandler(async () => {
          await pending;
          return getGetV1OnboardingStateResponseMock({
            completedSteps: ["ONBOARDING_COMPLETE"],
          });
        }),
      );
      const { result, rerender, queryClient } = renderGate();
      fireEvent[interaction](document.body, { key: "a" });
      await act(async () => releaseOnboarding());
      await waitFor(() => expect(queryClient.isFetching()).toBe(0));
      expect(result.current.isOpen).toBe(false);
      context.pathname = "/team";
      rerender();
      await waitFor(() => expect(result.current.isOpen).toBe(true));
      fireEvent[interaction](document.body, { key: "a" });
      expect(result.current.isOpen).toBe(true);
    },
  );

  it("does not defer for a navigation click on the previous route", async () => {
    context.pathname = "/settings";
    const { result, rerender } = renderGate();
    fireEvent.pointerDown(document.body);
    context.pathname = "/copilot";
    rerender();
    await waitFor(() => expect(result.current.isOpen).toBe(true));
  });

  it("waits for server onboarding before showing an eligible user", async () => {
    const { result } = renderGate();
    expect(result.current.isOpen).toBe(false);
    await waitFor(() => expect(result.current.isOpen).toBe(true));
  });

  it.each(["hire-experts", "autogpt-new-layout"])(
    "stays closed with %s disabled",
    (flag) => {
      context.flags[flag] = false;
      expectDeferred();
    },
  );

  it("waits for feature flag readiness", () => {
    context.ready = false;
    expectDeferred();
  });

  it("waits for auth", () => {
    context.user = null;
    expectDeferred();
  });

  it.each(["2026-09-21T00:00:00Z", undefined, "invalid"])(
    "does not show for signup %s",
    (created_at) => {
      context.user = { id: "user-1", created_at };
      expectDeferred();
    },
  );

  it("defers while the caller has another dialog open", () => {
    expectDeferred(false);
  });

  it.each([
    "sessionId=session-1",
    "expertId=expert-1",
    "kickoff=1",
    "autosubmit=true",
    "modal=skills",
  ])("does not interrupt %s or its handoff", (search) => {
    context.search = search;
    const { result, rerender, queryClient } = renderGate();
    expect(result.current.isOpen).toBe(false);
    context.search = "";
    rerender();
    expect(result.current.isOpen).toBe(false);
    expect(queryClient.isFetching()).toBe(0);
  });

  it("stays closed when onboarding cannot be verified", async () => {
    server.use(
      http.get("/api/proxy/api/onboarding", () => HttpResponse.error()),
    );
    const { result, queryClient } = renderGate();
    await waitFor(() => expect(queryClient.isFetching()).toBe(0));
    expect(result.current.isOpen).toBe(false);
  });

  it.each<{ completedSteps: OnboardingStep[] }>([
    { completedSteps: [] },
    { completedSteps: ["ONBOARDING_COMPLETE", "WORKFLOWS_MOVED"] },
  ])(
    "stays closed for completed steps $completedSteps",
    async ({ completedSteps }) => {
      mockOnboarding(completedSteps);
      const { result, queryClient } = renderGate();
      await waitFor(() => expect(queryClient.isFetching()).toBe(0));
      expect(result.current.isOpen).toBe(false);
    },
  );
});

describe("migration notice dismissal", () => {
  it("retains dismissal after remount when local storage cannot be written", async () => {
    vi.spyOn(window.localStorage, "setItem").mockImplementation(() => {
      throw new Error("Storage unavailable");
    });
    const { result, queryClient, unmount } = renderGate();
    await waitFor(() => expect(result.current.isOpen).toBe(true));
    act(() => result.current.dismiss());
    await waitFor(() => expect(queryClient.isMutating()).toBe(0));
    unmount();
    expect(renderGate(true, queryClient).result.current.isOpen).toBe(false);
  });

  it("closes immediately and persists exactly the migration step", async () => {
    const savedSteps: (string | null)[] = [];
    server.use(
      getPostV1CompleteOnboardingStepMockHandler(
        ({ request }: { request: Request }) => {
          savedSteps.push(new URL(request.url).searchParams.get("step"));
        },
      ),
    );
    const { result, unmount } = renderGate();
    await waitFor(() => expect(result.current.isOpen).toBe(true));
    act(() => result.current.dismiss());
    expect(result.current.isOpen).toBe(false);
    await waitFor(() => expect(savedSteps).toEqual(["WORKFLOWS_MOVED"]));
    unmount();
    expect(renderGate().result.current.isOpen).toBe(false);
  });

  it("stays dismissed when storage and the server are unavailable", async () => {
    vi.spyOn(window.localStorage, "setItem").mockImplementation(() => {
      throw new Error("Storage unavailable");
    });
    server.use(
      http.post("/api/proxy/api/onboarding/step", () =>
        HttpResponse.json({}, { status: 503 }),
      ),
    );
    const { result, rerender, queryClient } = renderGate();
    await waitFor(() => expect(result.current.isOpen).toBe(true));
    act(() => result.current.dismiss());
    expect(result.current.isOpen).toBe(false);
    await waitFor(() => expect(queryClient.isMutating()).toBe(0));
    rerender();
    expect(result.current.isOpen).toBe(false);
  });

  it("does not reuse another account's cached onboarding", async () => {
    const { result, rerender, queryClient } = renderGate();
    await waitFor(() => expect(result.current.isOpen).toBe(true));
    act(() => result.current.dismiss());
    mockOnboarding(["ONBOARDING_COMPLETE", "WORKFLOWS_MOVED"]);
    context.user = { id: "user-2", created_at: "2026-09-01T00:00:00Z" };
    rerender();
    expect(result.current.isOpen).toBe(false);
    await waitFor(() => expect(queryClient.isFetching()).toBe(0));
    expect(result.current.isOpen).toBe(false);
    mockOnboarding();
    context.user = { id: "user-3", created_at: "2026-09-01T00:00:00Z" };
    rerender();
    expect(result.current.isOpen).toBe(false);
    await waitFor(() => expect(result.current.isOpen).toBe(true));
  });
});
