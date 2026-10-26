import { act, render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { motionPreference } = vi.hoisted(() => ({
  motionPreference: { reduced: false },
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
import { AmbientPrompts } from "../AmbientPrompts";

const OPERATIONS = BRAIN_DUMP_PROMPT_SETS.Operations;

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
});

afterEach(() => {
  vi.useRealTimers();
});

describe("AmbientPrompts", () => {
  it("is hidden from assistive tech and has nothing to focus or click", () => {
    render(<AmbientPrompts role="Operations" screen="rest" />);

    expect(layer().getAttribute("aria-hidden")).toBe("true");
    expect(layer().className).toContain("pointer-events-none");
    expect(
      layer().querySelectorAll("button, a, input, textarea, [tabindex]"),
    ).toHaveLength(0);
  });

  it("shows four of the role's questions and both examples as quotes", () => {
    render(<AmbientPrompts role="Operations" screen="rest" />);

    expect(shownQuestions()).toEqual([0, 1, 2, 3]);
    expect(screen.getAllByText("For example")).toHaveLength(2);
    for (const example of OPERATIONS.examples) {
      expect(screen.getByText(`“${example}”`)).toBeDefined();
    }
  });

  it("swaps one question card every 7 seconds, in turn", async () => {
    render(<AmbientPrompts role="Operations" screen="rest" />);

    await act(async () => {
      vi.advanceTimersByTime(6_999);
    });
    expect(shownQuestions()).toEqual([0, 1, 2, 3]);

    await act(async () => {
      vi.advanceTimersByTime(1);
    });
    expect(shownQuestions()).toEqual([1, 2, 3, 4]);

    await act(async () => {
      vi.advanceTimersByTime(7_000);
    });
    expect(shownQuestions()).toEqual([2, 3, 4, 5]);
    // The quotes stay put.
    expect(screen.getAllByText("For example")).toHaveLength(2);
  });

  it("rotates half as often when motion is reduced", async () => {
    motionPreference.reduced = true;
    render(<AmbientPrompts role="Operations" screen="rest" />);

    await act(async () => {
      vi.advanceTimersByTime(7_000);
    });
    expect(shownQuestions()).toEqual([0, 1, 2, 3]);

    await act(async () => {
      vi.advanceTimersByTime(7_000);
    });
    expect(shownQuestions()).toEqual([1, 2, 3, 4]);
  });

  it("uses the Other set for a role it does not know", () => {
    render(<AmbientPrompts role="Dentist" screen="rest" />);

    expect(
      screen.getByText(BRAIN_DUMP_PROMPT_SETS.Other.questions[0].text),
    ).toBeDefined();
  });

  it.each(["typing", "recording"] as const)(
    "steps back while %s",
    (screenState) => {
      render(<AmbientPrompts role="Operations" screen={screenState} />);

      expect(layer().className).toContain("opacity-50");
    },
  );

  it.each(["recovery", "failed", "insufficient"] as const)(
    "is gone on the %s screen",
    (screenState) => {
      render(<AmbientPrompts role="Operations" screen={screenState} />);

      expect(screen.queryByTestId("ambient-prompts")).toBeNull();
    },
  );
});
