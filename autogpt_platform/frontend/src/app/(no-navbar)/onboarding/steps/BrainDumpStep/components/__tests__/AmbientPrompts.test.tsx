import { act, fireEvent, render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { motionPreference, device } = vi.hoisted(() => ({
  motionPreference: { reduced: false },
  device: { isWide: true },
}));

vi.mock("../../useIsWideLayout", () => ({
  useIsWideLayout: () => device.isWide,
}));

// Exit animations keep the outgoing card mounted until a frame that fake
// timers never paint; these tests are about which card is current.
vi.mock("framer-motion", async (importOriginal) => ({
  ...(await importOriginal<typeof import("framer-motion")>()),
  AnimatePresence: function AnimatePresence({
    children,
  }: {
    children: ReactNode;
  }) {
    return <>{children}</>;
  },
  useReducedMotion: () => motionPreference.reduced,
}));

import { BRAIN_DUMP_PROMPT_SETS } from "../../prompts";
import type { ScreenState } from "../../useBrainDumpStep";
import { AmbientPrompts } from "../AmbientPrompts";

const OPERATIONS = BRAIN_DUMP_PROMPT_SETS.Operations;
const onTogglePause = vi.fn();

function renderLayer(role: string, screenState: ScreenState, isPaused = false) {
  return render(
    <AmbientPrompts
      role={role}
      screen={screenState}
      isPaused={isPaused}
      onTogglePause={onTogglePause}
    />,
  );
}

function shownExamples() {
  return OPERATIONS.examples
    .map((example, index) => ({ index, text: `“${example}”` }))
    .filter(({ text }) => screen.queryByText(text) !== null)
    .map(({ index }) => index);
}

function layer() {
  return screen.getByTestId("ambient-prompts");
}

function shownQuestions() {
  return OPERATIONS.questions
    .map((question, index) => ({ index, text: question.text }))
    .filter(({ text }) => screen.queryByText(text) !== null)
    .map(({ index }) => index);
}

beforeEach(() => {
  vi.useFakeTimers();
  motionPreference.reduced = false;
  device.isWide = true;
  onTogglePause.mockReset();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("AmbientPrompts", () => {
  it("is hidden from assistive tech and has nothing to focus", () => {
    renderLayer("Operations", "rest");

    expect(layer().getAttribute("aria-hidden")).toBe("true");
    expect(
      layer().querySelectorAll("button, a, input, textarea, [tabindex]"),
    ).toHaveLength(0);
  });

  it("pauses or resumes on a click on any card", () => {
    renderLayer("Operations", "rest");

    fireEvent.click(screen.getByText(OPERATIONS.questions[0].text));
    fireEvent.click(screen.getByText(`“${OPERATIONS.examples[0]}”`));

    expect(onTogglePause).toHaveBeenCalledTimes(2);
  });

  it("shows four of the role's questions and two examples as quotes", () => {
    renderLayer("Operations", "rest");

    expect(shownQuestions()).toEqual([0, 1, 2, 3]);
    expect(screen.getAllByText("For example")).toHaveLength(2);
    expect(shownExamples()).toEqual([0, 1]);
  });

  it("swaps one card every 3.5 seconds, hopping across the page", async () => {
    renderLayer("Operations", "rest");

    await act(async () => {
      vi.advanceTimersByTime(3_499);
    });
    expect(shownQuestions()).toEqual([0, 1, 2, 3]);

    // Card 3 (right, top) first, then card 1 (left, middle)...
    await act(async () => {
      vi.advanceTimersByTime(1);
    });
    expect(shownQuestions()).toEqual([0, 1, 3, 4]);
    await act(async () => {
      vi.advanceTimersByTime(3_500);
    });
    expect(shownQuestions()).toEqual([0, 3, 4, 5]);
    expect(shownExamples()).toEqual([0, 1]);

    // ...then a quote (right, bottom), so the examples take turns too.
    await act(async () => {
      vi.advanceTimersByTime(3_500);
    });
    expect(shownExamples()).toEqual([0, 2]);
  });

  it("rotates half as often when motion is reduced", async () => {
    motionPreference.reduced = true;
    renderLayer("Operations", "rest");

    await act(async () => {
      vi.advanceTimersByTime(3_500);
    });
    expect(shownQuestions()).toEqual([0, 1, 2, 3]);

    await act(async () => {
      vi.advanceTimersByTime(3_500);
    });
    expect(shownQuestions()).toEqual([0, 1, 3, 4]);
  });

  it("uses the Other set for a role it does not know", () => {
    renderLayer("Dentist", "rest");

    expect(
      screen.getByText(BRAIN_DUMP_PROMPT_SETS.Other.questions[0].text),
    ).toBeDefined();
  });

  it.each(["typing", "recording"] as const)(
    "steps back while %s",
    (screenState) => {
      renderLayer("Operations", screenState);

      expect(layer().className).toContain("opacity-50");
    },
  );

  it("keeps changing while recording", async () => {
    renderLayer("Operations", "recording");

    await act(async () => {
      vi.advanceTimersByTime(3_500);
    });

    expect(shownQuestions()).toEqual([0, 1, 3, 4]);
  });

  it("holds still while paused", async () => {
    renderLayer("Operations", "recording", true);

    await act(async () => {
      vi.advanceTimersByTime(60_000);
    });

    expect(shownQuestions()).toEqual([0, 1, 2, 3]);
    expect(layer().getAttribute("data-paused")).toBe("true");
  });

  it("holds still while the dump is being sent", async () => {
    renderLayer("Operations", "processing");
    const before = shownQuestions();

    await act(async () => {
      vi.advanceTimersByTime(60_000);
    });

    expect(shownQuestions()).toEqual(before);
  });

  it("runs no timer where the layer does not fit", () => {
    device.isWide = false;
    renderLayer("Operations", "rest");

    expect(vi.getTimerCount()).toBe(0);
  });

  it.each(["recovery", "failed", "insufficient"] as const)(
    "is gone on the %s screen",
    (screenState) => {
      renderLayer("Operations", screenState);

      expect(screen.queryByTestId("ambient-prompts")).toBeNull();
    },
  );
});
