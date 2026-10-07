import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import type { ToolUIPart } from "ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CopilotChatActionsProvider } from "../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import { ExpertOnboardingCard } from "../ExpertOnboardingCard";
import { PendingOnboardingContext } from "../PendingOnboardingContext";

const EXPERT_ID = "1a5b1a10-6d10-4d7c-9d0d-2f6f1d9c0f11";
const CALL_ID = "call-onboarding-1";

vi.mock("../../../useExpertMap", () => ({
  useExpertMap: () => ({
    expertsById: new Map([
      [
        EXPERT_ID,
        {
          id: EXPERT_ID,
          name: "Ada",
          avatarUrl: null,
          role: "Social & Content Repurposing",
          isArchived: false,
          readOnlyReason: null,
        },
      ],
    ]),
  }),
}));

afterEach(() => {
  cleanup();
  sessionStorage.clear();
});

function onboardingPart(overrides: Record<string, unknown> = {}): ToolUIPart {
  return {
    type: "tool-expert_onboarding",
    toolCallId: CALL_ID,
    state: "output-available",
    input: {},
    output: {
      type: "expert_onboarding",
      message: "Which outcome first?; Which services?",
      session_id: "session-1",
      expert_id: EXPERT_ID,
      greeting: "Hi, I'm Ada — good to be working with you.",
      steps: [
        {
          question: "Which outcome should I start with?",
          keyword: "outcome",
          options: ["Social listening", "Campaign briefs"],
        },
        {
          question: "Which service should I be connected to?",
          keyword: "service",
          options: ["Linear", "GitHub"],
        },
      ],
      ...overrides,
    },
  } as unknown as ToolUIPart;
}

function createSendMock() {
  return vi.fn<(message: string) => Promise<void>>();
}

function cardTree(
  part: ToolUIPart,
  pendingCallId: string | null,
  onSend: ReturnType<typeof createSendMock>,
  rowKey: string,
) {
  return (
    <CopilotChatActionsProvider onSend={onSend}>
      <PendingOnboardingContext.Provider value={pendingCallId}>
        <ExpertOnboardingCard key={rowKey} part={part} />
      </PendingOnboardingContext.Provider>
    </CopilotChatActionsProvider>
  );
}

function renderCard(
  part: ToolUIPart,
  pendingCallId: string | null = CALL_ID,
  onSend = createSendMock(),
) {
  const { rerender } = render(cardTree(part, pendingCallId, onSend, "stream"));
  // The settled turn swaps the streamed message for its saved copy, which
  // carries a different id — React sees a new key and remounts the card.
  function rekeyRow() {
    rerender(cardTree(part, pendingCallId, onSend, "session-1-seq-2"));
  }
  return { onSend, rekeyRow };
}

function actionButton(label: string): HTMLButtonElement {
  return screen.getByLabelText(label) as HTMLButtonElement;
}

describe("ExpertOnboardingCard", () => {
  it("greets the user under the expert's identity with the shared area label", () => {
    renderCard(onboardingPart());

    expect(screen.getByText("Ada")).toBeDefined();
    expect(screen.getByText("Social Media Manager")).toBeDefined();
    expect(screen.queryByText("Social & Content Repurposing")).toBeNull();
    expect(
      screen.getByText("Hi, I'm Ada — good to be working with you."),
    ).toBeDefined();
  });

  it("shows one question at a time with tappable options", () => {
    renderCard(onboardingPart());

    expect(
      screen.getByText("Which outcome should I start with?"),
    ).toBeDefined();
    expect(
      screen.queryByText("Which service should I be connected to?"),
    ).toBeNull();
    expect(screen.getAllByRole("radio")).toHaveLength(2);
    expect(screen.getByText("1 of 2")).toBeDefined();
  });

  it("waits for Next after an option is tapped", () => {
    renderCard(onboardingPart());

    expect(actionButton("Next question").disabled).toBe(true);

    fireEvent.click(screen.getByRole("radio", { name: /Social listening/ }));

    // The tap only selects, leaving room to change or edit the answer.
    expect(screen.getByText("1 of 2")).toBeDefined();
    expect(actionButton("Next question").disabled).toBe(false);

    fireEvent.click(actionButton("Next question"));
    expect(
      screen.getByText("Which service should I be connected to?"),
    ).toBeDefined();
  });

  it("merges edited options into one typed answer", () => {
    renderCard(onboardingPart());

    fireEvent.click(
      screen.getByRole("button", { name: "Edit Social listening" }),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Edit Campaign briefs" }),
    );

    expect((screen.getByRole("textbox") as HTMLTextAreaElement).value).toBe(
      "Social listening\nCampaign briefs",
    );
    expect(screen.getAllByRole("radio")).toHaveLength(2);
    expect(screen.getByText("1 of 2")).toBeDefined();
  });

  it("will not send until the last step is answered", () => {
    renderCard(onboardingPart());

    fireEvent.click(screen.getByRole("radio", { name: /Social listening/ }));
    fireEvent.click(actionButton("Next question"));

    expect(actionButton("Send answers").disabled).toBe(true);

    fireEvent.click(screen.getByRole("radio", { name: /Linear/ }));

    expect(actionButton("Send answers").disabled).toBe(false);
  });

  it("keeps the Next button for a typed answer, which cannot auto-advance", () => {
    renderCard(onboardingPart());

    fireEvent.click(screen.getByRole("button", { name: /Type something/ }));
    const textarea = screen.getByRole("textbox");

    expect(actionButton("Next question").disabled).toBe(true);

    fireEvent.change(textarea, { target: { value: "Competitor tracking" } });

    // Still on step one — there is no single moment a typed answer is done.
    expect(screen.getByText("1 of 2")).toBeDefined();
    fireEvent.click(actionButton("Next question"));
    expect(screen.getByText("2 of 2")).toBeDefined();
  });

  it("sends every answer as one message on the last step", () => {
    const { onSend } = renderCard(onboardingPart());

    fireEvent.click(screen.getByRole("radio", { name: /Social listening/ }));
    fireEvent.click(actionButton("Next question"));

    expect(screen.getByText("2 of 2")).toBeDefined();
    fireEvent.click(screen.getByRole("radio", { name: /Linear/ }));
    fireEvent.click(actionButton("Send answers"));

    expect(onSend).toHaveBeenCalledTimes(1);
    const message = onSend.mock.calls[0][0] as string;
    expect(message).toContain("Which outcome should I start with?");
    expect(message).toContain("Social listening");
    expect(message).toContain("Which service should I be connected to?");
    expect(message).toContain("Linear");
  });

  it("lets the user step back to change an answer", () => {
    renderCard(onboardingPart());

    fireEvent.click(screen.getByRole("radio", { name: /Social listening/ }));
    fireEvent.click(actionButton("Next question"));
    fireEvent.click(actionButton("Previous question"));

    expect(screen.getByText("1 of 2")).toBeDefined();
    expect(
      screen
        .getByRole("radio", { name: /Social listening/ })
        .getAttribute("aria-checked"),
    ).toBe("true");
  });

  it("skips the whole card without answering", () => {
    const { onSend } = renderCard(onboardingPart());

    fireEvent.click(screen.getByRole("button", { name: "Skip" }));

    expect(onSend).toHaveBeenCalledWith(
      "Let's skip the setup questions for now.",
    );
  });

  it("renders as history once the thread has moved past it", () => {
    renderCard(onboardingPart(), null);

    expect(screen.queryByRole("radio")).toBeNull();
    expect(screen.queryByRole("button", { name: "Skip" })).toBeNull();
    expect(screen.getByText(/Setup questions from/)).toBeDefined();
  });

  it("shows a pending line while the card is still streaming", () => {
    const part = {
      type: "tool-expert_onboarding",
      toolCallId: CALL_ID,
      state: "input-streaming",
      input: {},
    } as unknown as ToolUIPart;

    renderCard(part);

    expect(screen.queryByRole("radio")).toBeNull();
  });

  it("reports a settled call whose output carries no usable steps", () => {
    renderCard(onboardingPart({ steps: [] }));

    expect(screen.queryByRole("radio")).toBeNull();
    expect(screen.queryByText("Ada")).toBeNull();
    // The call is over, so this must not sit on the loading line forever.
    expect(screen.getByText(/Couldn.t open the setup questions/)).toBeDefined();
  });

  it("keeps its place when the settled turn re-keys its message", () => {
    const { rekeyRow } = renderCard(onboardingPart());

    fireEvent.click(screen.getByRole("radio", { name: /Social listening/ }));
    fireEvent.click(actionButton("Next question"));
    expect(screen.getByText("2 of 2")).toBeDefined();

    rekeyRow();

    expect(screen.getByText("2 of 2")).toBeDefined();
    expect(
      screen.getByText("Which service should I be connected to?"),
    ).toBeDefined();
    fireEvent.click(actionButton("Previous question"));
    expect(
      screen
        .getByRole("radio", { name: /Social listening/ })
        .getAttribute("aria-checked"),
    ).toBe("true");
  });

  it("restores the answers given so far after a reload", () => {
    renderCard(onboardingPart());
    fireEvent.click(screen.getByRole("radio", { name: /Social listening/ }));
    fireEvent.click(actionButton("Next question"));
    fireEvent.click(screen.getByRole("radio", { name: /Linear/ }));
    cleanup();

    renderCard(onboardingPart());

    expect(screen.getByText("2 of 2")).toBeDefined();
    expect(
      screen
        .getByRole("radio", { name: /Linear/ })
        .getAttribute("aria-checked"),
    ).toBe("true");
  });

  it("starts a different card from the beginning", () => {
    renderCard(onboardingPart());
    fireEvent.click(screen.getByRole("radio", { name: /Social listening/ }));
    fireEvent.click(actionButton("Next question"));
    cleanup();

    const other = { ...onboardingPart(), toolCallId: "call-onboarding-2" };
    renderCard(other as ToolUIPart, "call-onboarding-2");

    expect(screen.getByText("1 of 2")).toBeDefined();
    expect(screen.queryByRole("radio", { checked: true })).toBeNull();
  });

  it("forgets its saved place once the answers have gone out", async () => {
    const onSend = createSendMock();
    onSend.mockResolvedValue(undefined);
    renderCard(onboardingPart(), CALL_ID, onSend);

    fireEvent.click(screen.getByRole("radio", { name: /Social listening/ }));
    fireEvent.click(actionButton("Next question"));
    fireEvent.click(screen.getByRole("radio", { name: /Linear/ }));
    fireEvent.click(actionButton("Send answers"));
    await waitFor(() =>
      expect(screen.getByText(/Setup questions from/)).toBeDefined(),
    );
    cleanup();

    renderCard(onboardingPart());

    expect(screen.getByText("1 of 2")).toBeDefined();
    expect(screen.queryByRole("radio", { checked: true })).toBeNull();
  });

  it("keeps the answers on screen when the send fails", async () => {
    const onSend = createSendMock();
    onSend.mockRejectedValue(new Error("no session"));
    renderCard(onboardingPart(), CALL_ID, onSend);

    fireEvent.click(screen.getByRole("radio", { name: /Social listening/ }));
    fireEvent.click(actionButton("Next question"));
    fireEvent.click(screen.getByRole("radio", { name: /Linear/ }));
    fireEvent.click(actionButton("Send answers"));

    await waitFor(() => expect(onSend).toHaveBeenCalledTimes(1));
    // Still the live form, with the answer intact — not a settled history row.
    await waitFor(() =>
      expect(
        screen
          .getByRole("radio", { name: /Linear/ })
          .getAttribute("aria-checked"),
      ).toBe("true"),
    );
    expect(screen.queryByText(/Setup questions from/)).toBeNull();
  });
});
