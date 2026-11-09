import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { expect, test } from "vitest";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { expectTypingKept } from "@/tests/integrations/typing";
import { CancelTrialFeedbackDialog } from "../CancelTrialFeedbackDialog";
import { CheckoutAbandonedDialog } from "../CheckoutAbandonedDialog";
import { NpsCard } from "../NpsCard";
import { OnboardingSkipPrompt } from "../OnboardingSkipPrompt";
import { OttoMistakeCard } from "../OttoMistakeCard";
import { TopUpDismissedCard } from "../TopUpDismissedCard";
import { WorkCheckCard } from "../WorkCheckCard";
import { NoInitialFocus } from "../components/NoInitialFocus";
import { BRAIN_DUMP_SKIP_REASONS, SAMPLE_STARTERS } from "../helpers";

function FocusForm() {
  const [note, setNote] = useState("");
  return (
    <>
      <NoInitialFocus />
      <textarea
        aria-label="Note"
        value={note}
        onChange={(e) => setNote(e.target.value)}
      />
    </>
  );
}

test("typing keeps focus after the prompt has opened", async () => {
  render(<FocusForm />);
  const note = screen.getByRole("textbox", {
    name: "Note",
  }) as HTMLTextAreaElement;
  await waitFor(() => expect(document.activeElement?.tagName).toBe("SPAN"));
  await expectTypingKept(note, "My feedback");
  await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
  expect(document.activeElement).toBe(note);
  await expectTypingKept(note, "useful ", 3);
});

test("trial cancellation collects a note and confirms the choice", async () => {
  render(<CancelTrialFeedbackDialog />);
  expect(
    screen.getByRole("group", {
      name: "Before you go, what's the main reason?",
    }),
  ).toBeDefined();
  await userEvent.click(screen.getByRole("button", { name: "Something else" }));
  const note = screen.getByRole("textbox", {
    name: "What would have made you stay?",
  }) as HTMLTextAreaElement;
  await expectTypingKept(note, "More integrations");
  await userEvent.click(screen.getByRole("button", { name: "Continue" }));
  expect(await screen.findByText("Cancel your trial?")).toBeDefined();
  await userEvent.click(screen.getByRole("button", { name: "Cancel trial" }));
  await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
});

test.each([{ starters: SAMPLE_STARTERS.developer }, { starters: [] }])(
  "trial cancellation offers help with stored tasks or no tasks",
  async ({ starters }) => {
    render(<CancelTrialFeedbackDialog starters={starters} />);
    await userEvent.click(
      screen.getByRole("button", { name: "I haven't seen results yet" }),
    );
    await userEvent.click(screen.getByRole("button", { name: "Continue" }));
    expect(await screen.findByText("Let Otto show you first")).toBeDefined();
    if (starters.length) {
      for (const starter of starters)
        expect(screen.getByRole("button", { name: starter })).toBeDefined();
    } else {
      expect(
        screen.getByText(
          "Tell me one weekly task and I'll do it with you now.",
        ),
      ).toBeDefined();
    }
    await userEvent.click(
      screen.getByRole("button", { name: "No thanks, cancel" }),
    );
    await userEvent.click(screen.getByRole("button", { name: "Keep trial" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  },
);

test("a trial cancellation can skip the feedback step", async () => {
  render(<CancelTrialFeedbackDialog />);
  expect(
    (screen.getByRole("button", { name: "Continue" }) as HTMLButtonElement)
      .disabled,
  ).toBe(true);
  await userEvent.click(screen.getByRole("button", { name: "Skip" }));
  expect(await screen.findByText("Cancel your trial?")).toBeDefined();
});

test.each(["trial", "upgrade"] as const)(
  "abandoned %s checkout accepts a reason and thanks the user",
  async (kind) => {
    render(<CheckoutAbandonedDialog kind={kind} />);
    await userEvent.click(
      screen.getByRole("button", { name: "Something else" }),
    );
    const note = screen.getByRole("textbox", {
      name: "Tell us a bit more (optional)",
    }) as HTMLTextAreaElement;
    await expectTypingKept(note, "Need more time");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText(/Thanks\./)).toBeDefined();
    expect(screen.queryByRole("button", { name: "Send" })).toBeNull();
  },
);

test.each([
  [3, "What's the biggest thing holding it back?"],
  [8, "What would make it a 10?"],
  [10, "What do you like most?"],
] as const)(
  "score %s shows the right follow-up and accepts a note",
  async (score, question) => {
    render(<NpsCard />);
    expect(
      screen.getByRole("group", {
        name: /How likely are you to recommend AutoGPT.*0.*10/,
      }),
    ).toBeDefined();
    await userEvent.click(screen.getByRole("button", { name: String(score) }));
    expect(
      screen
        .getByRole("button", { name: String(score) })
        .getAttribute("aria-pressed"),
    ).toBe("true");
    const note = screen.getByRole("textbox", {
      name: question,
    }) as HTMLTextAreaElement;
    await expectTypingKept(note, "Useful work");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText(/Thanks, we read every one/)).toBeDefined();
  },
);

test("NPS can be dismissed", async () => {
  render(<NpsCard />);
  await userEvent.click(screen.getByRole("button", { name: "Close" }));
  expect(screen.queryByRole("region")).toBeNull();
});

test("skipping onboarding accepts one reason", async () => {
  render(
    <OnboardingSkipPrompt
      question="Mind telling us why?"
      reasons={BRAIN_DUMP_SKIP_REASONS}
    />,
  );
  expect(
    screen.getByRole("group", { name: "Mind telling us why?" }),
  ).toBeDefined();
  await userEvent.click(
    screen.getByRole("button", { name: "I'd rather type" }),
  );
  expect(await screen.findByText("Got it, thanks.")).toBeDefined();
});

test("a mistake prompt accepts a reason and a note", async () => {
  render(<OttoMistakeCard />);
  await userEvent.click(screen.getByRole("button", { name: "Something else" }));
  const note = screen.getByRole("textbox", {
    name: "Tell us a bit more (optional)",
  }) as HTMLTextAreaElement;
  await expectTypingKept(note, "Wrong city");
  await userEvent.click(screen.getByRole("button", { name: "Send" }));
  expect(
    await screen.findByText(/The team reads every one of these/),
  ).toBeDefined();
});

test("a top-up prompt accepts a reason", async () => {
  render(<TopUpDismissedCard />);
  await userEvent.click(screen.getByRole("button", { name: "A lower price" }));
  expect(
    screen
      .getByRole("button", { name: "A lower price" })
      .getAttribute("aria-pressed"),
  ).toBe("true");
  await userEvent.click(screen.getByRole("button", { name: "Send" }));
  expect(await screen.findByText(/Thanks, we read every one/)).toBeDefined();
});

test.each(["Not good", "Okay", "Great"])(
  "work check handles %s feedback",
  async (rating) => {
    render(
      <WorkCheckCard
        name="Maria"
        avatarUrl={null}
        note="Quick check-in on this task"
      />,
    );
    await userEvent.click(screen.getByRole("button", { name: rating }));
    if (rating === "Great") {
      expect(
        await screen.findByText(/We'll pass that on to Maria/),
      ).toBeDefined();
    } else {
      await userEvent.click(
        screen.getByRole("button", { name: "Got facts wrong" }),
      );
      await userEvent.click(screen.getByRole("button", { name: "Send" }));
      expect(await screen.findByText(/Maria learns from this/)).toBeDefined();
    }
  },
);
