import { act, render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import type { ToolUIPart } from "ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const capture = vi.hoisted(() => vi.fn());
vi.mock("posthog-js", () => ({ default: { capture } }));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: { id: "user-1" }, isLoggedIn: true }),
}));

const onboarding = vi.hoisted(() => ({
  current: {
    state: { completedSteps: [] as string[] } as {
      completedSteps: string[];
    } | null,
    completeStep: vi.fn(),
  },
}));
vi.mock("@/providers/onboarding/onboarding-provider", async (original) => ({
  ...(await original<object>()),
  useOnboarding: () => onboarding.current,
}));

vi.mock("@/services/usageExperience/useUsageExperience", () => ({
  useUsageExperience: () => ({ experience: { isActiveTrial: true } }),
}));

const sessionList = vi.hoisted(() => ({
  current: { sessions: [] as { id: string }[], isLoading: false },
}));
vi.mock("@/app/(platform)/copilot/useSessionList", () => ({
  useSessionList: () => sessionList.current,
}));

import { ExpertChangeGroup } from "@/app/(platform)/copilot/components/ToolChain/ExpertCards";
import { BudgetLowHint } from "@/app/(platform)/copilot/components/UsageLimits/BudgetLowHint";
import type { UsageSnapshot } from "@/services/usageExperience/useUsageExperience";
import { tourTarget } from "../../FirstRunTour/steps";
import { useFirstRunTourStore } from "../../FirstRunTour/store";
import { ContextualTips } from "../ContextualTips";
import { isFirstSecondChat } from "../helpers";
import { requestTip, useContextualTipStore } from "../store";
import { mentionsCost } from "../tips";
import { latestReplyMentionsCost } from "../useTipMoment";

function stubLayout() {
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(
    function (this: HTMLElement) {
      return this.hasAttribute("data-tour-target")
        ? DOMRect.fromRect({ x: 10, y: 100, width: 240, height: 40 })
        : DOMRect.fromRect({ x: 0, y: 0, width: 0, height: 0 });
    },
  );
}

function Page() {
  return (
    <div>
      <div {...tourTarget("chats")}>Recent chats</div>
      <div {...tourTarget("team")}>Team</div>
      <div {...tourTarget("wallet")}>Wallet</div>
      <label>
        Message
        <textarea />
      </label>
      <ContextualTips />
    </div>
  );
}

beforeEach(() => {
  window.localStorage.clear();
  capture.mockReset();
  onboarding.current = { state: { completedSteps: [] }, completeStep: vi.fn() };
  sessionList.current = { sessions: [], isLoading: false };
  act(() => useContextualTipStore.setState({ requested: [] }));
  stubLayout();
});

afterEach(() => {
  act(() => useFirstRunTourStore.getState().close());
  vi.restoreAllMocks();
});

describe("contextual tips — when a cost comes up", () => {
  it.each([
    "That costs a few cents per image.",
    "This run uses about 2 credits.",
    "Each image is 1 credit.",
    "It comes out of your trial allowance, not your card.",
    "Automation credits pay for scheduled runs.",
  ])("counts Otto's billing words: %s", (text) => {
    expect(mentionsCost(text)).toBe(true);
  });

  it.each([
    "I can cut your shipping costs by batching orders.",
    "Your $500 marketing budget is enough for two campaigns.",
    "Update the credit card on file in Stripe.",
    "We take credit cards and bank transfers.",
    "Give credit to the photographer in the caption.",
  ])("ignores money talk that is about the business: %s", (text) => {
    expect(mentionsCost(text)).toBe(false);
  });

  it("only reads Otto's latest finished reply", () => {
    const reply = (text: string) => ({
      role: "assistant",
      parts: [{ type: "text", text }],
    });
    const cost = reply("That costs a few cents per image.");

    expect(latestReplyMentionsCost([cost], false)).toBe(true);
    expect(latestReplyMentionsCost([cost], true)).toBe(false);
    expect(
      latestReplyMentionsCost([cost, { role: "user", parts: [] }], false),
    ).toBe(false);
    expect(latestReplyMentionsCost([cost, reply("Done.")], false)).toBe(false);
  });

  it("treats running low on the allowance as the moment too", () => {
    const usage = {
      budget: { state: "low", window: "trial", percentUsed: 85 },
      experience: { isLifetimeTrial: true, trialPercent: 85 },
    } as unknown as UsageSnapshot;

    render(<BudgetLowHint usage={usage} />);

    expect(useContextualTipStore.getState().requested).toEqual(["first-cost"]);
  });
});

describe("contextual tips — when an expert is proposed", () => {
  function proposal(kind: "raise" | "hire"): ToolUIPart {
    return {
      type: "tool-raise_expert",
      state: "output-available",
      toolCallId: "call-1",
      input: {},
      output: {
        type: "expert_change_proposed",
        applied: false,
        confirmation_id: "c-1",
        preview: { kind, name: "Fiona", role: "Bookkeeper" },
      },
    } as ToolUIPart;
  }

  it("asks for the tip when Otto proposes an expert", () => {
    render(<ExpertChangeGroup parts={[proposal("hire")]} />);

    expect(useContextualTipStore.getState().requested).toEqual([
      "first-expert",
    ]);
  });

  it("stays quiet on a read-only transcript", () => {
    render(<ExpertChangeGroup parts={[proposal("raise")]} readOnly />);

    expect(useContextualTipStore.getState().requested).toEqual([]);
  });
});

describe("contextual tips — the first second chat", () => {
  it("counts going from one chat to two, and nothing else", () => {
    expect(isFirstSecondChat(1, 2)).toBe(true);
    expect(isFirstSecondChat(null, 2)).toBe(false);
    expect(isFirstSecondChat(0, 1)).toBe(false);
    expect(isFirstSecondChat(5, 6)).toBe(false);
  });

  it("points at the chat list when a second chat appears", async () => {
    sessionList.current = { sessions: [{ id: "a" }], isLoading: false };
    const { rerender } = render(<Page />);
    expect(screen.queryByTestId("contextual-tip")).toBeNull();

    sessionList.current = {
      sessions: [{ id: "b" }, { id: "a" }],
      isLoading: false,
    };
    rerender(<Page />);

    expect(await screen.findByText("You have two chats now")).toBeDefined();
  });

  it("says nothing to someone who already has several chats", async () => {
    sessionList.current = {
      sessions: [{ id: "a" }, { id: "b" }, { id: "c" }],
      isLoading: false,
    };
    const { rerender } = render(<Page />);
    sessionList.current = {
      sessions: [{ id: "d" }, { id: "a" }, { id: "b" }, { id: "c" }],
      isLoading: false,
    };
    rerender(<Page />);

    await act(async () => {});
    expect(screen.queryByTestId("contextual-tip")).toBeNull();
  });
});

describe("contextual tips — showing one", () => {
  it("shows beside its target without taking focus, and is gone for good after Got it", async () => {
    render(<Page />);
    const textarea = screen.getByRole("textbox", { name: "Message" });
    textarea.focus();

    act(() => requestTip("first-expert"));

    expect(await screen.findByText("Experts can own tasks")).toBeDefined();
    expect(document.activeElement).toBe(textarea);

    await userEvent.click(screen.getByRole("button", { name: "Got it" }));

    await waitFor(() =>
      expect(screen.queryByTestId("contextual-tip")).toBeNull(),
    );
    expect(onboarding.current.completeStep).toHaveBeenCalledWith("EXPERT_TIP");
    expect(
      window.localStorage.getItem("autogpt:contextual-tip-seen:first-expert"),
    ).toBe("user-1");

    act(() => requestTip("first-expert"));
    await act(async () => {});
    expect(screen.queryByTestId("contextual-tip")).toBeNull();
  });

  it("says where costs come from on a trial", async () => {
    render(<Page />);
    act(() => requestTip("first-cost"));

    expect(
      await screen.findByText(
        "Chats and experts use your trial allowance, and automations use credits. Check both here.",
      ),
    ).toBeDefined();
  });

  it("never shows a tip this user already has on another device", async () => {
    onboarding.current.state = { completedSteps: ["CHATS_TIP"] };
    render(<Page />);

    act(() => requestTip("second-chat"));

    await act(async () => {});
    expect(screen.queryByTestId("contextual-tip")).toBeNull();
    expect(useContextualTipStore.getState().requested).toEqual([]);
  });

  it("waits while the walkthrough is open, then shows", async () => {
    render(<Page />);
    act(() => useFirstRunTourStore.getState().open("menu"));
    act(() => requestTip("first-cost"));

    await act(async () => {});
    expect(screen.queryByTestId("contextual-tip")).toBeNull();

    act(() => useFirstRunTourStore.getState().close());
    expect(await screen.findByText("Where costs come from")).toBeDefined();
  });

  it("shows one tip at a time, oldest first", async () => {
    render(<Page />);
    act(() => {
      requestTip("first-expert");
      requestTip("first-cost");
    });

    expect(await screen.findByText("Experts can own tasks")).toBeDefined();
    expect(screen.queryByText("Where costs come from")).toBeNull();

    await userEvent.click(screen.getByRole("button", { name: "Got it" }));

    expect(await screen.findByText("Where costs come from")).toBeDefined();
  });

  it("puts the tip away with Escape", async () => {
    render(<Page />);
    act(() => requestTip("second-chat"));
    await screen.findByText("You have two chats now");

    await userEvent.keyboard("{Escape}");

    await waitFor(() =>
      expect(screen.queryByTestId("contextual-tip")).toBeNull(),
    );
    expect(onboarding.current.completeStep).toHaveBeenCalledWith("CHATS_TIP");
  });
});
