import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { trackBrainDump } = vi.hoisted(() => ({ trackBrainDump: vi.fn() }));

vi.mock("@/services/onboarding/brain-dump-analytics", () => ({
  trackBrainDump,
}));

// The swap animation keeps the outgoing question mounted for a moment;
// these tests are about which question is current, not the transition.
vi.mock("@/components/atoms/SwapFade/SwapFade", () => ({
  SwapFade: function SwapFade({ children }: { children: ReactNode }) {
    return <div>{children}</div>;
  },
}));

import { BRAIN_DUMP_PROMPT_SETS } from "../../prompts";
import { usePromptRotation } from "../../usePromptRotation";
import { OttoMightAsk } from "../OttoMightAsk";

const QUESTIONS = BRAIN_DUMP_PROMPT_SETS["Sales/BD"].questions;

function LivePrompt() {
  const prompt = usePromptRotation({ role: "Sales/BD", screen: "recording" });
  return (
    <OttoMightAsk
      question={prompt.question.text}
      onAnother={() => prompt.advance("click")}
      announcement={prompt.announcement}
    />
  );
}

beforeEach(() => {
  trackBrainDump.mockClear();
});

describe("OttoMightAsk", () => {
  it("shows the role's question under its label", () => {
    render(<LivePrompt />);

    expect(screen.getByText("Otto might ask")).toBeDefined();
    expect(screen.getByText(QUESTIONS[4].text)).toBeDefined();
  });

  it("moves to the next question when asked for another", async () => {
    render(<LivePrompt />);

    await userEvent.click(
      screen.getByRole("button", { name: "Another question" }),
    );

    expect(await screen.findByText(QUESTIONS[5].text)).toBeDefined();
    expect(screen.queryByText(QUESTIONS[4].text)).toBeNull();
    expect(trackBrainDump).toHaveBeenCalledWith("brain_dump_prompt_advanced", {
      via: "click",
      screen: "recording",
      prompt_index: 5,
      prompt_set: "Sales/BD",
    });
  });

  it("announces a question only once the user asks for it", async () => {
    render(<LivePrompt />);
    const region = screen.getByTestId("otto-might-ask-announcer");

    expect(region.getAttribute("aria-live")).toBe("polite");
    expect(region.textContent).toBe("");

    await userEvent.click(
      screen.getByRole("button", { name: "Another question" }),
    );

    expect(region.textContent).toBe(`Otto might ask: ${QUESTIONS[5].text}`);
  });

  it("offers no button where the question is only a hint", () => {
    render(<OttoMightAsk question={QUESTIONS[0].text} compact />);

    expect(screen.queryByRole("button")).toBeNull();
    expect(screen.getByText(QUESTIONS[0].text)).toBeDefined();
  });
});
