import {
  getGetV1OnboardingStateMockHandler,
  getGetV1OnboardingStateResponseMock,
} from "@/app/api/__generated__/endpoints/onboarding/onboarding.msw";
import { server } from "@/mocks/mock-server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import type { ComponentType } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const context = vi.hoisted(() => ({
  user: { id: "returning-user", created_at: "2026-09-01T00:00:00Z" },
  pathname: "/library",
  search: "",
  flags: {} as Record<string, boolean>,
  readiness: {} as Record<string, boolean>,
  state: null as { completedSteps: string[] } | null,
  noticeWait: null as Promise<void> | null,
  completeStep: vi.fn(),
  capture: vi.fn(),
  getState: vi.fn(),
  postedStep: vi.fn(),
  push: vi.fn(),
}));
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: context.user }),
}));
vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  useOnboarding: () => ({
    state: context.state,
    completeStep: context.completeStep,
  }),
}));
vi.mock("posthog-js", () => ({ default: { capture: context.capture } }));
vi.mock("next/navigation", () => ({
  usePathname: () => context.pathname,
  useSearchParams: () => new URLSearchParams(context.search),
  useRouter: () => ({ push: context.push }),
}));
vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {
    ONBOARDING_BRAIN_DUMP: "onboarding-brain-dump",
    HIRE_EXPERTS: "hire-experts",
    AUTOGPT_NEW_LAYOUT: "autogpt-new-layout",
  },
  useGetFlag: (flag: string) => context.flags[flag] ?? false,
  useFlagStatus: (flag: string) => ({
    enabled: context.flags[flag] ?? false,
    ready: context.readiness[flag] ?? true,
  }),
}));

import { AgentsTabIntro } from "@/app/(platform)/library/components/AgentsTabIntro/AgentsTabIntro";
import { MarketplaceTabIntro } from "@/app/(platform)/marketplace/components/MarketplaceTabIntro/MarketplaceTabIntro";
import { BuildTabIntro } from "@/app/(platform)/build/components/BuildTabIntro/BuildTabIntro";
import { WorkflowsMovedNotice } from "../WorkflowsMovedNotice";

function renderTogether(Intro: ComponentType = AgentsTabIntro) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const introPath = context.pathname;
  function Fixture() {
    return (
      <QueryClientProvider client={queryClient}>
        {context.pathname === introPath && <Intro />}
        <WorkflowsMovedNotice />
      </QueryClientProvider>
    );
  }
  const rendered = render(<Fixture />);
  return {
    ...rendered,
    refresh: () => rendered.rerender(<Fixture />),
    queryClient,
  };
}

function expectOnlyDialog(name: string) {
  expect(screen.getByRole("dialog", { name })).toBeTruthy();
  expect(screen.queryAllByRole("dialog", { hidden: true })).toHaveLength(1);
}

function expectNoWrites() {
  expect(context.completeStep).not.toHaveBeenCalled();
  expect(context.postedStep).not.toHaveBeenCalled();
}

beforeEach(() => {
  vi.restoreAllMocks();
  vi.clearAllMocks();
  vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
  window.localStorage.clear();
  window.sessionStorage.clear();
  context.user = { id: "returning-user", created_at: "2026-09-01T00:00:00Z" };
  context.pathname = "/library";
  context.search = "";
  context.flags = {
    "onboarding-brain-dump": true,
    "hire-experts": true,
    "autogpt-new-layout": true,
  };
  context.readiness = {};
  context.state = { completedSteps: ["ONBOARDING_COMPLETE"] };
  context.noticeWait = null;
  server.use(
    getGetV1OnboardingStateMockHandler(async () => {
      context.getState();
      await context.noticeWait;
      return getGetV1OnboardingStateResponseMock({
        completedSteps: ["ONBOARDING_COMPLETE"],
      });
    }),
    http.post("/api/proxy/api/onboarding/step", ({ request }) => {
      context.postedStep(new URL(request.url).searchParams.get("step"));
      return new HttpResponse(null, { status: 200 });
    }),
  );
});

describe("migration notice with production first-visit tutorials", () => {
  it("replaces the unseen agents tutorial, including after dismissal and remount", async () => {
    const mounted = renderTogether();
    await screen.findByRole("dialog", { name: "Your agents have moved" });
    expectOnlyDialog("Your agents have moved");
    expectNoWrites();
    expect(context.capture).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole("button", { name: "Got it" }));
    await waitFor(() =>
      expect(context.postedStep).toHaveBeenCalledExactlyOnceWith(
        "WORKFLOWS_MOVED",
      ),
    );
    expect(screen.queryAllByRole("dialog", { hidden: true })).toHaveLength(0);
    mounted.unmount();
    renderTogether();
    expect(screen.queryAllByRole("dialog", { hidden: true })).toHaveLength(0);
    expect(context.completeStep).not.toHaveBeenCalled();
    expect(
      window.localStorage.getItem("autogpt:tab-intro-seen:agents"),
    ).toBeNull();
    expect(context.capture).not.toHaveBeenCalled();
  });

  it.each(["flags", "intro record"])(
    "never flashes the agents tutorial when %s arrives first",
    async (first) => {
      context.state = null;
      context.readiness = {
        "hire-experts": false,
        "autogpt-new-layout": false,
      };
      const { refresh } = renderTogether();
      expect(screen.queryAllByRole("dialog", { hidden: true })).toHaveLength(0);
      if (first === "flags") {
        context.readiness = {};
        refresh();
        await screen.findByRole("dialog", { name: "Your agents have moved" });
        expectOnlyDialog("Your agents have moved");
        context.state = { completedSteps: ["ONBOARDING_COMPLETE"] };
      } else {
        context.state = { completedSteps: ["ONBOARDING_COMPLETE"] };
        refresh();
        expect(screen.queryAllByRole("dialog", { hidden: true })).toHaveLength(
          0,
        );
        context.readiness = {};
      }
      refresh();
      await screen.findByRole("dialog", { name: "Your agents have moved" });
      expectOnlyDialog("Your agents have moved");
      expect(context.capture).not.toHaveBeenCalled();
      expectNoWrites();
    },
  );

  it("keeps both dialogs closed while the migration record is delayed", async () => {
    let release = () => {};
    context.noticeWait = new Promise<void>((resolve) => {
      release = resolve;
    });
    renderTogether();
    await waitFor(() => expect(context.getState).toHaveBeenCalledTimes(1));
    expect(screen.queryAllByRole("dialog", { hidden: true })).toHaveLength(0);
    expectNoWrites();
    await act(async () => release());
    await screen.findByRole("dialog", { name: "Your agents have moved" });
    expectOnlyDialog("Your agents have moved");
    expect(context.capture).not.toHaveBeenCalled();
  });

  it("preserves the agents tutorial for users who joined after the rollout", async () => {
    context.user.created_at = "2026-09-22T00:00:00Z";
    renderTogether();
    expectOnlyDialog("Your mission control.");
    expectNoWrites();
    await userEvent.click(
      screen.getByRole("button", { name: "See my agents" }),
    );
    await waitFor(() =>
      expect(screen.queryAllByRole("dialog", { hidden: true })).toHaveLength(0),
    );
    expect(context.completeStep).toHaveBeenCalledExactlyOnceWith(
      "AGENTS_TAB_INTRO",
    );
    expect(context.getState).not.toHaveBeenCalled();
    expect(context.postedStep).not.toHaveBeenCalled();
  });

  it.each(["hire-experts", "autogpt-new-layout"])(
    "preserves the old tutorial when delayed %s resolves off",
    async (flag) => {
      context.flags[flag] = false;
      context.readiness[flag] = false;
      const { refresh } = renderTogether();
      expect(screen.queryAllByRole("dialog", { hidden: true })).toHaveLength(0);
      context.readiness[flag] = true;
      refresh();
      expectOnlyDialog("Your mission control.");
      expectNoWrites();
      await userEvent.click(
        screen.getByRole("button", { name: "See my agents" }),
      );
      expect(context.completeStep).toHaveBeenCalledExactlyOnceWith(
        "AGENTS_TAB_INTRO",
      );
      expect(context.getState).not.toHaveBeenCalled();
      expect(context.postedStep).not.toHaveBeenCalled();
    },
  );

  it.each([
    [
      "/marketplace",
      MarketplaceTabIntro,
      "Agents ready to work.",
      "Browse featured agents",
      "MARKETPLACE_TAB_INTRO",
    ],
    [
      "/build",
      BuildTabIntro,
      "Create your own workflows.",
      "Ask Otto to build it",
      "BUILD_TAB_INTRO",
    ],
  ] as const)(
    "preserves the actual tutorial on %s without showing migration",
    async (path, Intro, title, cta, step) => {
      context.pathname = path;
      renderTogether(Intro);
      expectOnlyDialog(title);
      expectNoWrites();
      await userEvent.click(screen.getByRole("button", { name: cta }));
      await waitFor(() =>
        expect(screen.queryAllByRole("dialog", { hidden: true })).toHaveLength(
          0,
        ),
      );
      expect(context.completeStep).toHaveBeenCalledExactlyOnceWith(step);
      expect(context.getState).not.toHaveBeenCalled();
      expect(context.postedStep).not.toHaveBeenCalled();
    },
  );

  it("does not interrupt the Build tutorial's explicit Ask Otto handoff", async () => {
    context.pathname = "/build";
    const { refresh, queryClient } = renderTogether(BuildTabIntro);
    await userEvent.click(
      screen.getByRole("button", { name: "Ask Otto to build it" }),
    );
    expect(context.push).toHaveBeenCalledExactlyOnceWith("/copilot?new=1");
    const destination = new URL(
      context.push.mock.calls[0][0],
      "http://localhost",
    );
    context.pathname = destination.pathname;
    context.search = destination.search;
    refresh();
    await waitFor(() => expect(queryClient.isFetching()).toBe(0));
    await act(async () => {
      await new Promise<void>((resolve) =>
        requestAnimationFrame(() => resolve()),
      );
    });
    expect(screen.queryAllByRole("dialog", { hidden: true })).toHaveLength(0);
    context.search = "";
    refresh();
    await act(async () => {
      await new Promise<void>((resolve) =>
        requestAnimationFrame(() => resolve()),
      );
    });
    expect(screen.queryAllByRole("dialog", { hidden: true })).toHaveLength(0);
    expect(context.postedStep).not.toHaveBeenCalled();
    context.pathname = "/team";
    refresh();
    await screen.findByRole("dialog", { name: "Your agents have moved" });
    expectOnlyDialog("Your agents have moved");
  });
});
