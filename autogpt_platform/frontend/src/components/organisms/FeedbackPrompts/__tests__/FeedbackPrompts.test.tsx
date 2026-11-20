import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { expect, test, vi } from "vitest";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { expectTypingKept } from "@/tests/integrations/typing";
import { CancelTrialFeedbackDialog } from "../CancelTrialFeedbackDialog";
import { CheckoutAbandonedDialog } from "../CheckoutAbandonedDialog";
import { NpsCard } from "../NpsCard";
import { OttoMistakeCard } from "../OttoMistakeCard";
import { TopUpQuestion } from "../TopUpQuestion";
import { WorkCheckCard } from "../WorkCheckCard";
import { ThanksLine } from "../components/ThanksLine";
import { CHANGELOG_BASE_URL } from "@/components/molecules/ChangelogPopup/changelog-constants";
import { NoInitialFocus } from "../components/NoInitialFocus";

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

test("trial cancellation asks for an optional reason on the confirm itself", async () => {
  const onConfirm = vi.fn();
  render(<CancelTrialFeedbackDialog onConfirm={onConfirm} />);
  expect(screen.getByText("Cancel your trial?")).toBeDefined();
  expect(
    screen.getByRole("group", { name: "What's the main reason? (optional)" }),
  ).toBeDefined();
  expect(screen.getByRole("button", { name: "Keep trial" })).toBeDefined();
  await userEvent.click(screen.getByRole("button", { name: "Cancel trial" }));
  expect(onConfirm).toHaveBeenCalledWith({ reason: null, note: "" });
  await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
});

test("trial cancellation sends the picked reason and its note", async () => {
  const onReason = vi.fn();
  const onConfirm = vi.fn();
  render(
    <CancelTrialFeedbackDialog onReason={onReason} onConfirm={onConfirm} />,
  );
  await userEvent.click(screen.getByRole("button", { name: "Something else" }));
  expect(onReason).toHaveBeenCalledWith("other");
  const note = screen.getByRole("textbox", {
    name: "Tell us a bit more (optional)",
  }) as HTMLTextAreaElement;
  await expectTypingKept(note, "More integrations");
  expect(screen.getByRole("button", { name: "Keep trial" })).toBeDefined();
  await userEvent.click(screen.getByRole("button", { name: "Cancel trial" }));
  expect(onConfirm).toHaveBeenCalledWith({
    reason: "other",
    note: "More integrations",
  });
});

test("keeping the trial sends nothing to cancel", async () => {
  const onConfirm = vi.fn();
  render(<CancelTrialFeedbackDialog onConfirm={onConfirm} />);
  await userEvent.click(
    screen.getByRole("button", { name: "I haven't seen results yet" }),
  );
  await userEvent.click(screen.getByRole("button", { name: "Keep trial" }));
  await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  expect(onConfirm).not.toHaveBeenCalled();
});

test("abandoned upgrade checkout accepts a reason and thanks the user", async () => {
  render(<CheckoutAbandonedDialog onTryAnotherCard={() => {}} />);
  await userEvent.click(screen.getByRole("button", { name: "Something else" }));
  const note = screen.getByRole("textbox", {
    name: "Tell us a bit more (optional)",
  }) as HTMLTextAreaElement;
  await expectTypingKept(note, "Need more time");
  await userEvent.click(screen.getByRole("button", { name: "Send" }));
  expect(await screen.findByText(/Thanks, we read every one/)).toBeDefined();
  expect(screen.queryByRole("button", { name: "Send" })).toBeNull();
  expect(screen.queryByRole("button", { name: "Try another card" })).toBeNull();
});

test("only a card answer offers another card", async () => {
  const onTryAnotherCard = vi.fn();
  render(<CheckoutAbandonedDialog onTryAnotherCard={onTryAnotherCard} />);
  await userEvent.click(
    screen.getByRole("button", { name: "My card didn't go through" }),
  );
  await userEvent.click(screen.getByRole("button", { name: "Send" }));
  await userEvent.click(
    await screen.findByRole("button", { name: "Try another card" }),
  );
  expect(onTryAnotherCard).toHaveBeenCalledOnce();
});

test.each([
  [3, "What's the main thing we should fix?"],
  [8, "What would make it a 10?"],
  [10, "What do you like most?"],
] as const)(
  "score %s shows the right follow-up and accepts a note",
  async (score, question) => {
    render(<NpsCard />);
    expect(
      screen.getByRole("group", {
        name: /recommend AutoGPT to a friend or colleague.*0.*10/,
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

test("a mistake prompt shares the work check's reasons", async () => {
  render(<OttoMistakeCard />);
  for (const name of ["I had to step in too much", "Not good enough to use"])
    expect(screen.getByRole("button", { name })).toBeDefined();
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

test("a top-up question accepts a reason", async () => {
  render(<TopUpQuestion />);
  await userEvent.click(screen.getByRole("button", { name: "A lower price" }));
  expect(
    screen
      .getByRole("button", { name: "A lower price" })
      .getAttribute("aria-pressed"),
  ).toBe("true");
  await userEvent.click(screen.getByRole("button", { name: "Send" }));
  expect(await screen.findByText(/Thanks, we read every one/)).toBeDefined();
});

test.each([
  ["Not good", "What went wrong?"],
  ["Okay", "What would have made it great?"],
  ["Great", null],
] as const)("work check handles %s feedback", async (rating, followUp) => {
  render(
    <WorkCheckCard
      name="Maria"
      avatarUrl={null}
      note="Quick check-in on this task"
    />,
  );
  await userEvent.click(screen.getByRole("button", { name: rating }));
  if (followUp === null) {
    expect(
      await screen.findByText(/We'll pass that on to Maria/),
    ).toBeDefined();
  } else {
    expect(screen.getByRole("group", { name: followUp })).toBeDefined();
    await userEvent.click(
      screen.getByRole("button", { name: "I had to step in too much" }),
    );
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(
      await screen.findByText("Thanks. The team reads every one of these."),
    ).toBeDefined();
  }
});

test("reason callbacks carry stable values and the note", async () => {
  const onReason = vi.fn();
  const onSubmit = vi.fn();
  render(<TopUpQuestion onReason={onReason} onSubmit={onSubmit} />);
  await userEvent.click(screen.getByRole("button", { name: "A lower price" }));
  expect(onReason).toHaveBeenCalledWith("lower_price");
  await userEvent.click(screen.getByRole("button", { name: "Something else" }));
  expect(onReason).toHaveBeenLastCalledWith("other");
  await userEvent.type(
    screen.getByRole("textbox", { name: "Tell us a bit more (optional)" }),
    "Less per month",
  );
  await userEvent.click(screen.getByRole("button", { name: "Send" }));
  expect(onSubmit).toHaveBeenCalledWith({
    reason: "other",
    note: "Less per month",
  });
});

test("closing a card without answering reports a dismissal", async () => {
  const onDismiss = vi.fn();
  render(<NpsCard onDismiss={onDismiss} />);
  await userEvent.click(screen.getByRole("button", { name: "Close" }));
  expect(onDismiss).toHaveBeenCalledOnce();
});

test("the thanks link opens the published changelog", () => {
  render(<ThanksLine />);
  expect(
    screen
      .getByRole("link", { name: "See what we've changed lately" })
      .getAttribute("href"),
  ).toBe(CHANGELOG_BASE_URL);
});
