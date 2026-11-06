import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const motion = vi.hoisted(() => ({ reduced: false }));

vi.mock("framer-motion", async (importOriginal) => ({
  ...(await importOriginal<typeof import("framer-motion")>()),
  useReducedMotion: () => motion.reduced,
}));

import { InsufficientState } from "../InsufficientState";

const QUESTION = "What do you want off your plate?";

interface Overrides {
  mode?: "voice" | "typed";
  heardText?: string | null;
  role?: string;
  canRecord?: boolean;
}

function renderScreen(overrides: Overrides = {}) {
  const handlers = { onTalk: vi.fn(), onWrite: vi.fn(), onSkip: vi.fn() };
  render(
    <InsufficientState
      mode={overrides.mode ?? "voice"}
      heardText={
        overrides.heardText === undefined
          ? "Um, okay. Hi. Is this working?"
          : overrides.heardText
      }
      role={overrides.role ?? "Founder/CEO"}
      canRecord={overrides.canRecord ?? true}
      {...handlers}
    />,
  );
  return handlers;
}

function buttonNames() {
  return screen.getAllByRole("button").map((button) => button.textContent);
}

function dots() {
  return screen.queryByTestId("otto-typing-dots");
}

beforeEach(() => {
  motion.reduced = false;
});

afterEach(() => {
  vi.useRealTimers();
});

describe("InsufficientState — what Otto heard", () => {
  it("shows the words heard and answers loud and clear", () => {
    renderScreen();

    expect(screen.getByText("You")).toBeDefined();
    expect(screen.getByText("Um, okay. Hi. Is this working?")).toBeDefined();
    expect(screen.getByText("Otto")).toBeDefined();
    expect(screen.getByText("Loud and clear! So tell me,")).toBeDefined();
    expect(screen.getByRole("heading", { name: QUESTION })).toBeDefined();
    expect(buttonNames()).toEqual(["Talk", "Write", "Continue without it"]);
  });

  it("says no words came through when nothing was heard", () => {
    renderScreen({ heardText: null });

    expect(screen.getByText("No words came through")).toBeDefined();
    expect(
      screen.getByText(
        "I couldn't hear any words. Check that your mic is on, then tell me,",
      ),
    ).toBeDefined();
    expect(screen.getByRole("heading", { name: QUESTION })).toBeDefined();
    expect(buttonNames()).toEqual(["Talk", "Write", "Continue without it"]);
  });

  it("treats a blank preview as no words", () => {
    renderScreen({ heardText: "   " });

    expect(screen.getByText("No words came through")).toBeDefined();
  });

  it("shows typed text and leads back to writing", () => {
    renderScreen({ mode: "typed", heardText: "idk" });

    expect(screen.getByText("idk")).toBeDefined();
    expect(
      screen.getByText("No problem. Let's start with an easy one."),
    ).toBeDefined();
    expect(screen.getByRole("heading", { name: QUESTION })).toBeDefined();
    expect(buttonNames()).toEqual([
      "Write it",
      "Talk instead",
      "Continue without it",
    ]);
  });

  it.each([
    { mode: "voice" as const, write: "Write" },
    { mode: "typed" as const, write: "Write it" },
  ])(
    "offers no Talk while the microphone is blocked ($mode)",
    ({ mode, write }) => {
      renderScreen({ mode, canRecord: false });

      expect(buttonNames()).toEqual([write, "Continue without it"]);
    },
  );
});

describe("InsufficientState — the example", () => {
  it("follows the role picked on the step before", () => {
    renderScreen({ role: "Sales/BD" });

    expect(
      screen.getByText(
        "“Building lead lists. It takes me half of every Monday.”",
      ),
    ).toBeDefined();
    expect(screen.getByText(/^Like:/)).toBeDefined();
  });

  it("falls back to Other for a role it does not know", () => {
    renderScreen({ role: "Dentist" });

    expect(
      screen.getByText("“The reminders and invoices for my students.”"),
    ).toBeDefined();
  });
});

describe("InsufficientState — actions", () => {
  it("talks, writes and continues without it through the handlers", async () => {
    const handlers = renderScreen();

    await userEvent.click(screen.getByRole("button", { name: "Talk" }));
    await userEvent.click(screen.getByRole("button", { name: "Write" }));
    await userEvent.click(
      screen.getByRole("button", { name: "Continue without it" }),
    );

    expect(handlers.onTalk).toHaveBeenCalledTimes(1);
    expect(handlers.onWrite).toHaveBeenCalledTimes(1);
    expect(handlers.onSkip).toHaveBeenCalledTimes(1);
  });

  it("maps the typed case's buttons to the same handlers", async () => {
    const handlers = renderScreen({ mode: "typed", heardText: "idk" });

    await userEvent.click(screen.getByRole("button", { name: "Write it" }));
    await userEvent.click(screen.getByRole("button", { name: "Talk instead" }));

    expect(handlers.onWrite).toHaveBeenCalledTimes(1);
    expect(handlers.onTalk).toHaveBeenCalledTimes(1);
  });
});

describe("InsufficientState — Otto typing", () => {
  it("shows typing dots, then his message, which is in the DOM all along", () => {
    vi.useFakeTimers();
    renderScreen();

    const message = screen.getByTestId("otto-reply-message");
    expect(dots()).not.toBeNull();
    expect(dots()?.getAttribute("aria-hidden")).toBe("true");
    expect(
      within(message).getByText("Loud and clear! So tell me,"),
    ).toBeDefined();
    expect(message.className).toContain("opacity-0");

    act(() => {
      vi.advanceTimersByTime(900);
    });

    expect(dots()).toBeNull();
    expect(message.className).not.toContain("opacity-0");
  });

  it("shows no dots and no fade with reduced motion", () => {
    motion.reduced = true;
    renderScreen();

    const message = screen.getByTestId("otto-reply-message");
    expect(dots()).toBeNull();
    expect(message.className).not.toContain("opacity-0");
    expect(message.className).not.toContain("transition");
    expect(screen.getByRole("heading", { name: QUESTION })).toBeDefined();
  });

  it("announces Otto's message to screen readers at once", () => {
    renderScreen();

    const reply = screen.getByRole("alert");
    expect(
      within(reply).getByText("Loud and clear! So tell me,"),
    ).toBeDefined();
    expect(
      within(reply).getByRole("heading", { name: QUESTION }),
    ).toBeDefined();
  });
});
