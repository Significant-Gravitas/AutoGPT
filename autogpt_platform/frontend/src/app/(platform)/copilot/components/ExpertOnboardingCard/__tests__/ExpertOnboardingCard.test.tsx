import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import type { ToolUIPart, UIDataTypes, UIMessage, UITools } from "ai";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { PendingAnswerContexts } from "../../ChatMessagesContainer/components/PendingAnswerContexts";
import { CopilotChatActionsProvider } from "../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import { AnsweredQuestionsContext } from "../../QuestionDock/AnsweredQuestionsContext";
import { ExpertOnboardingCard } from "../ExpertOnboardingCard";
import { buildOnboardingAnswersMessage } from "../helpers";
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

function questionNumber() {
  return screen.getByRole("progressbar").getAttribute("aria-valuetext");
}

function isChecked(name: RegExp) {
  return screen.getByRole("radio", { name }).getAttribute("aria-checked");
}

// A tap moves on by itself after a short beat.
async function answerFirstStep() {
  fireEvent.click(screen.getByRole("radio", { name: /Social listening/ }));
  await screen.findByText("Which service should I be connected to?");
}

// The last question moves on to the review, not to Send.
async function answerLastStep() {
  fireEvent.click(screen.getByRole("radio", { name: /Linear/ }));
  await screen.findByText("Check your answers");
}

async function answerAllSteps() {
  await answerFirstStep();
  await answerLastStep();
}

function multiSelectPart() {
  return onboardingPart({
    steps: [
      {
        question: "Which services should I be connected to?",
        keyword: "services",
        options: ["Linear", "GitHub", "Figma"],
        allow_multiple: true,
      },
      {
        question: "Which outcome should I start with?",
        keyword: "outcome",
        options: ["Social listening", "Campaign briefs"],
      },
    ],
  });
}

describe("ExpertOnboardingCard", () => {
  it("leaves the expert's name and greeting to the thread header", () => {
    renderCard(onboardingPart());

    expect(screen.queryByText("Ada")).toBeNull();
    expect(
      screen.queryByText("Hi, I'm Ada — good to be working with you."),
    ).toBeNull();
  });

  it("shows one question at a time with lettered options and Other", () => {
    renderCard(onboardingPart());

    expect(
      screen.getByText("Which outcome should I start with?"),
    ).toBeDefined();
    expect(
      screen.queryByText("Which service should I be connected to?"),
    ).toBeNull();
    expect(screen.getAllByRole("radio")).toHaveLength(2);
    expect(screen.getByRole("button", { name: "Other" })).toBeDefined();
    expect(questionNumber()).toBe("Question 1 of 2");
  });

  it("moves to the next question once an option is tapped", async () => {
    renderCard(onboardingPart());

    await answerFirstStep();

    expect(questionNumber()).toBe("Question 2 of 2");
  });

  it("picks an option by its letter key", async () => {
    renderCard(onboardingPart());

    fireEvent.keyDown(screen.getByRole("radio", { name: /Social listening/ }), {
      key: "b",
    });

    expect(isChecked(/Campaign briefs/)).toBe("true");
    await screen.findByText("Which service should I be connected to?");
  });

  it("moves the selection with arrow keys without leaving the question", () => {
    renderCard(onboardingPart());

    fireEvent.keyDown(screen.getByRole("radio", { name: /Social listening/ }), {
      key: "ArrowDown",
    });

    expect(isChecked(/Campaign briefs/)).toBe("true");
    expect(questionNumber()).toBe("Question 1 of 2");
  });

  it("shows every answer for review before anything is sent", async () => {
    const { onSend } = renderCard(onboardingPart());
    await answerFirstStep();

    expect(actionButton("Review answers").disabled).toBe(true);

    await answerLastStep();

    expect(onSend).not.toHaveBeenCalled();
    expect(
      screen.getByRole("button", { name: /Which outcome.*Social listening/ }),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: /Which service.*Linear/ }),
    ).toBeDefined();
    expect(actionButton("Send answers").disabled).toBe(false);
  });

  it("jumps from the review to a question and straight back", async () => {
    renderCard(onboardingPart());
    await answerAllSteps();

    fireEvent.click(screen.getByRole("button", { name: /Which outcome/ }));

    expect(questionNumber()).toBe("Question 1 of 2");
    expect(isChecked(/Social listening/)).toBe("true");

    // The changed answer goes back to the review, not to question two.
    fireEvent.click(screen.getByRole("radio", { name: /Campaign briefs/ }));
    await screen.findByText("Check your answers");
    expect(
      screen.getByRole("button", { name: /Which outcome.*Campaign briefs/ }),
    ).toBeDefined();
  });

  it("keeps the Next button for a typed answer, which cannot auto-advance", () => {
    renderCard(onboardingPart());

    fireEvent.click(screen.getByRole("button", { name: "Other" }));
    const textarea = screen.getByRole("textbox");

    expect(actionButton("Next question").disabled).toBe(true);

    fireEvent.change(textarea, { target: { value: "Competitor tracking" } });

    // Still on step one — there is no single moment a typed answer is done.
    expect(questionNumber()).toBe("Question 1 of 2");
    fireEvent.click(actionButton("Next question"));
    expect(questionNumber()).toBe("Question 2 of 2");
  });

  it("sends every answer as one message from the review", async () => {
    const { onSend } = renderCard(onboardingPart());
    await answerAllSteps();

    fireEvent.click(actionButton("Send answers"));

    expect(onSend).toHaveBeenCalledTimes(1);
    const message = onSend.mock.calls[0][0] as string;
    expect(message).toContain("Which outcome should I start with?");
    expect(message).toContain("Social listening");
    expect(message).toContain("Which service should I be connected to?");
    expect(message).toContain("Linear");
  });

  it("lets the user step back to change an answer", async () => {
    renderCard(onboardingPart());
    await answerFirstStep();

    fireEvent.click(actionButton("Previous question"));

    expect(questionNumber()).toBe("Question 1 of 2");
    expect(isChecked(/Social listening/)).toBe("true");
  });

  it("skips the whole card without answering", () => {
    const { onSend } = renderCard(onboardingPart());

    fireEvent.click(screen.getByRole("button", { name: "Skip questions" }));

    expect(onSend).toHaveBeenCalledWith(
      "Let's skip the setup questions for now.",
    );
  });

  it("renders as history once the thread has moved past it", () => {
    renderCard(onboardingPart(), null);

    expect(screen.queryByRole("radio")).toBeNull();
    expect(screen.queryByRole("button", { name: "Skip questions" })).toBeNull();
    expect(screen.getByText(/Setup questions from/)).toBeDefined();
  });

  it("keeps each question and the answer given once it is history", () => {
    render(
      <CopilotChatActionsProvider onSend={createSendMock()}>
        <AnsweredQuestionsContext.Provider
          value={
            new Map([
              [
                CALL_ID,
                new Map([
                  ["outcome", "Campaign briefs"],
                  ["service", "Linear, GitHub"],
                ]),
              ],
            ])
          }
        >
          <PendingOnboardingContext.Provider value={null}>
            <ExpertOnboardingCard part={onboardingPart()} />
          </PendingOnboardingContext.Provider>
        </AnsweredQuestionsContext.Provider>
      </CopilotChatActionsProvider>,
    );

    expect(screen.queryByRole("radio")).toBeNull();
    expect(
      screen.getByText("Which outcome should I start with?"),
    ).toBeDefined();
    expect(screen.getByText("Campaign briefs")).toBeDefined();
    expect(
      screen.getByText("Which service should I be connected to?"),
    ).toBeDefined();
    expect(screen.getByText("Linear, GitHub")).toBeDefined();
  });

  it("shows the sent answers on a settled card and the form's on the review, and the review's once sent", async () => {
    const channelStep = {
      question: "Which channel should I post to?",
      keyword: "channel",
      options: ["LinkedIn", "Bluesky"],
      allow_multiple: false,
    };
    const earlier = {
      ...onboardingPart({ steps: [channelStep] }),
      toolCallId: "call-onboarding-earlier",
    } as ToolUIPart;
    render(
      <Chat
        initial={[
          chatMessage("assistant", [earlier]),
          chatMessage("user", [
            {
              type: "text",
              text: buildOnboardingAnswersMessage([channelStep], {
                channel: "Bluesky",
              }),
            },
          ]),
          chatMessage("assistant", [multiSelectPart()]),
        ]}
      />,
    );

    fireEvent.click(screen.getByRole("checkbox", { name: /Linear/ }));
    fireEvent.click(screen.getByRole("checkbox", { name: /GitHub/ }));
    fireEvent.click(actionButton("Next question"));
    fireEvent.click(screen.getByRole("radio", { name: /Campaign briefs/ }));
    await screen.findByText("Check your answers");

    expect(screen.getByText("Bluesky")).toBeDefined();
    expect(
      screen.getByRole("button", { name: /Which services.*Linear, GitHub/ }),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: /Which outcome.*Campaign briefs/ }),
    ).toBeDefined();

    fireEvent.click(actionButton("Send answers"));

    // Settled now, so these come back from the reply the send just added.
    expect(await screen.findByText("Linear, GitHub")).toBeDefined();
    expect(screen.queryByText("Check your answers")).toBeNull();
    expect(screen.getByText("Campaign briefs")).toBeDefined();
    expect(screen.getByText("Bluesky")).toBeDefined();
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

  it("keeps its place when the settled turn re-keys its message", async () => {
    const { rekeyRow } = renderCard(onboardingPart());
    await answerFirstStep();

    rekeyRow();

    expect(questionNumber()).toBe("Question 2 of 2");
    expect(
      screen.getByText("Which service should I be connected to?"),
    ).toBeDefined();
    fireEvent.click(actionButton("Previous question"));
    expect(isChecked(/Social listening/)).toBe("true");
  });

  it("still moves on when the row re-keys during the auto-advance beat", () => {
    const { rekeyRow } = renderCard(onboardingPart());

    fireEvent.click(screen.getByRole("radio", { name: /Social listening/ }));
    rekeyRow();

    expect(questionNumber()).toBe("Question 2 of 2");
    expect(
      screen.getByText("Which service should I be connected to?"),
    ).toBeDefined();
    fireEvent.click(actionButton("Previous question"));
    expect(isChecked(/Social listening/)).toBe("true");
  });

  it("restores the answers given so far after a reload", async () => {
    renderCard(onboardingPart());
    await answerFirstStep();
    fireEvent.click(screen.getByRole("radio", { name: /Linear/ }));
    cleanup();

    renderCard(onboardingPart());

    // The last pick was saved as a move to the review, so that is where the
    // reload lands, with both answers in place.
    expect(screen.getByText("Check your answers")).toBeDefined();
    expect(
      screen.getByRole("button", { name: /Which outcome.*Social listening/ }),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: /Which service.*Linear/ }),
    ).toBeDefined();
  });

  it("starts a different card from the beginning", async () => {
    renderCard(onboardingPart());
    await answerFirstStep();
    cleanup();

    const other = { ...onboardingPart(), toolCallId: "call-onboarding-2" };
    renderCard(other as ToolUIPart, "call-onboarding-2");

    expect(questionNumber()).toBe("Question 1 of 2");
    expect(screen.queryByRole("radio", { checked: true })).toBeNull();
  });

  it("forgets its saved place once the answers have gone out", async () => {
    const onSend = createSendMock();
    onSend.mockResolvedValue(undefined);
    renderCard(onboardingPart(), CALL_ID, onSend);

    await answerAllSteps();
    fireEvent.click(actionButton("Send answers"));
    await waitFor(() =>
      expect(screen.getByText(/Setup questions from/)).toBeDefined(),
    );
    cleanup();

    renderCard(onboardingPart());

    expect(questionNumber()).toBe("Question 1 of 2");
    expect(screen.queryByRole("radio", { checked: true })).toBeNull();
  });

  it("keeps the answers on screen when the send fails", async () => {
    const onSend = createSendMock();
    onSend.mockRejectedValue(new Error("no session"));
    renderCard(onboardingPart(), CALL_ID, onSend);

    await answerAllSteps();
    fireEvent.click(actionButton("Send answers"));

    await waitFor(() => expect(onSend).toHaveBeenCalledTimes(1));
    // Still the live form, with the answers intact — not a settled history row.
    await waitFor(() =>
      expect(actionButton("Send answers").disabled).toBe(false),
    );
    expect(
      screen.getByRole("button", { name: /Which service.*Linear/ }),
    ).toBeDefined();
    expect(screen.queryByText(/Setup questions from/)).toBeNull();
  });

  describe("multi-select questions", () => {
    it("ticks several options without moving on", async () => {
      renderCard(multiSelectPart());

      expect(screen.getAllByRole("checkbox")).toHaveLength(3);
      fireEvent.click(screen.getByRole("checkbox", { name: /Linear/ }));
      fireEvent.click(screen.getByRole("checkbox", { name: /Figma/ }));

      await new Promise((resolve) => setTimeout(resolve, 400));
      expect(questionNumber()).toBe("Question 1 of 2");
      expect(
        screen
          .getByRole("checkbox", { name: /Linear/ })
          .getAttribute("aria-checked"),
      ).toBe("true");
      expect(
        screen
          .getByRole("checkbox", { name: /Figma/ })
          .getAttribute("aria-checked"),
      ).toBe("true");

      fireEvent.click(actionButton("Next question"));
      expect(questionNumber()).toBe("Question 2 of 2");
    });

    it("sends the ticks and the typed text as a list", async () => {
      const { onSend } = renderCard(multiSelectPart());

      fireEvent.click(screen.getByRole("checkbox", { name: /Linear/ }));
      fireEvent.click(screen.getByRole("checkbox", { name: /GitHub/ }));
      fireEvent.click(screen.getByRole("button", { name: "Other" }));
      fireEvent.change(screen.getByRole("textbox"), {
        target: { value: "Notion" },
      });
      fireEvent.click(actionButton("Next question"));
      fireEvent.click(screen.getByRole("radio", { name: /Campaign briefs/ }));
      await screen.findByText("Check your answers");

      expect(
        screen.getByRole("button", {
          name: /Which services.*Linear, GitHub, Notion/,
        }),
      ).toBeDefined();

      fireEvent.click(actionButton("Send answers"));

      expect(onSend.mock.calls[0][0]).toContain(
        "> Which services should I be connected to?\n\n- Linear\n- GitHub\n- Notion",
      );
    });

    it("keeps the ticks after a reload", () => {
      renderCard(multiSelectPart());
      fireEvent.click(screen.getByRole("checkbox", { name: /GitHub/ }));
      cleanup();

      renderCard(multiSelectPart());

      expect(
        screen
          .getByRole("checkbox", { name: /GitHub/ })
          .getAttribute("aria-checked"),
      ).toBe("true");
    });
  });

  describe("keyboard", () => {
    it("puts the first question in focus", () => {
      renderCard(onboardingPart());

      expect(document.activeElement).toBe(
        screen.getByRole("radio", { name: /Social listening/ }),
      );
    });

    it("picks by letter from anywhere in the card", async () => {
      renderCard(onboardingPart());

      fireEvent.keyDown(
        screen.getByRole("button", { name: "Skip questions" }),
        {
          key: "b",
        },
      );

      expect(isChecked(/Campaign briefs/)).toBe("true");
      await screen.findByText("Which service should I be connected to?");
    });

    it("goes back with Backspace", async () => {
      renderCard(onboardingPart());
      await answerFirstStep();

      fireEvent.keyDown(screen.getByRole("radio", { name: /Linear/ }), {
        key: "Backspace",
      });

      expect(questionNumber()).toBe("Question 1 of 2");
    });

    it("skips with Escape", () => {
      const { onSend } = renderCard(onboardingPart());

      fireEvent.keyDown(
        screen.getByRole("radio", { name: /Social listening/ }),
        { key: "Escape" },
      );

      expect(onSend).toHaveBeenCalledWith(
        "Let's skip the setup questions for now.",
      );
    });

    it("leaves letters and Backspace to the Other text box", () => {
      const { onSend } = renderCard(onboardingPart());
      fireEvent.click(screen.getByRole("button", { name: "Other" }));
      const textarea = screen.getByRole("textbox");

      fireEvent.keyDown(textarea, { key: "a" });
      fireEvent.keyDown(textarea, { key: "Backspace" });
      fireEvent.keyDown(textarea, { key: "Escape" });

      expect(screen.queryByRole("radio", { checked: true })).toBeNull();
      expect(onSend).not.toHaveBeenCalled();
    });

    it("ignores keys while an IME is composing", () => {
      renderCard(onboardingPart());

      fireEvent.keyDown(
        screen.getByRole("radio", { name: /Social listening/ }),
        { key: "b", isComposing: true },
      );

      expect(screen.queryByRole("radio", { checked: true })).toBeNull();
    });

    it("moves focus to Send on the review", async () => {
      renderCard(onboardingPart());
      await answerAllSteps();

      expect(document.activeElement).toBe(actionButton("Send answers"));
    });
  });
});

type ChatMessage = UIMessage<unknown, UIDataTypes, UITools>;

function chatMessage(
  role: "user" | "assistant",
  parts: ChatMessage["parts"],
): ChatMessage {
  return { id: crypto.randomUUID(), role, parts };
}

// The thread's real contexts over a message list that a send appends to,
// as the chat does.
function Chat({ initial }: { initial: ChatMessage[] }) {
  const [messages, setMessages] = useState(initial);
  async function onSend(text: string) {
    setMessages((previous) => [
      ...previous,
      chatMessage("user", [{ type: "text", text }]),
    ]);
  }
  return (
    <CopilotChatActionsProvider onSend={onSend}>
      <PendingAnswerContexts messages={messages}>
        {messages.flatMap((message) =>
          message.parts.flatMap((part) =>
            part.type === "tool-expert_onboarding"
              ? [
                  <ExpertOnboardingCard
                    key={(part as ToolUIPart).toolCallId}
                    part={part as ToolUIPart}
                  />,
                ]
              : [],
          ),
        )}
      </PendingAnswerContexts>
    </CopilotChatActionsProvider>
  );
}
