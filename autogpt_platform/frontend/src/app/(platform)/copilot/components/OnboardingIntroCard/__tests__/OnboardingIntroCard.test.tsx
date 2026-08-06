import type { SuggestedPrompt } from "@/app/api/__generated__/models/suggestedPrompt";
import {
  act,
  cleanup,
  fireEvent,
  normalizeWhitespace,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  introRevealTimings,
  OnboardingIntroCard,
} from "../OnboardingIntroCard";

const toastSpy = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: (...args: unknown[]) => toastSpy(...(args as [])),
  useToast: () => ({ toast: toastSpy }),
}));

const PROMPTS: SuggestedPrompt[] = [
  {
    title: "Summarise my inbox",
    prompt: "Every morning, summarise my unread email and flag what matters.",
    icon: "envelope",
  },
  {
    title: "Watch competitor pricing",
    prompt: "Check competitor pricing pages weekly and tell me what moved.",
    icon: "not-a-real-icon-slug",
  },
];

const GREETING = "You spend most of your week chasing updates.";
const COPY_LABEL = "Copy your recording's transcript";
const FOOTER = "Want to do something else? Just write it in the textbox below.";

function writeTextMock(impl: () => Promise<void>) {
  const writeText = vi.fn(impl);
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    writable: true,
    value: { writeText },
  });
  return writeText;
}

beforeEach(() => {
  toastSpy.mockReset();
});

afterEach(() => {
  cleanup();
});

describe("OnboardingIntroCard — content", () => {
  it("greets the user by name and renders the greeting, prompts and footer", () => {
    const { container } = render(
      <OnboardingIntroCard
        name="Alex"
        greeting={GREETING}
        prompts={PROMPTS}
        onSelectPrompt={vi.fn()}
      />,
    );

    const text = normalizeWhitespace(container);
    expect(text).toContain("Hey, Alex");
    expect(text).toContain(GREETING);

    expect(screen.getByText("Summarise my inbox")).toBeDefined();
    expect(screen.getByText("Watch competitor pricing")).toBeDefined();
    expect(screen.getByText(FOOTER)).toBeDefined();
  });

  it("renders a row for a prompt whose icon slug the frontend does not know", () => {
    render(
      <OnboardingIntroCard
        name="Alex"
        greeting={GREETING}
        prompts={[PROMPTS[1]]}
        onSelectPrompt={vi.fn()}
      />,
    );

    expect(
      screen.getByRole("button", { name: "Watch competitor pricing" }),
    ).toBeDefined();
  });

  it("omits the prompt list entirely when the server sent no prompts", () => {
    render(
      <OnboardingIntroCard
        name="Alex"
        greeting={GREETING}
        prompts={[]}
        onSelectPrompt={vi.fn()}
      />,
    );

    expect(screen.queryByRole("list")).toBeNull();
    expect(screen.getByText(FOOTER)).toBeDefined();
  });
});

describe("OnboardingIntroCard — picking a prompt", () => {
  it("sends the full prompt, not the short title shown in the row", async () => {
    const onSelectPrompt = vi.fn();
    render(
      <OnboardingIntroCard
        name="Alex"
        greeting={GREETING}
        prompts={PROMPTS}
        onSelectPrompt={onSelectPrompt}
      />,
    );

    await userEvent.click(
      screen.getByRole("button", { name: "Summarise my inbox" }),
    );

    expect(onSelectPrompt).toHaveBeenCalledExactlyOnceWith(PROMPTS[0].prompt);
  });

  it("does not fire while the composer is disabled", async () => {
    const onSelectPrompt = vi.fn();
    render(
      <OnboardingIntroCard
        name="Alex"
        greeting={GREETING}
        prompts={PROMPTS}
        onSelectPrompt={onSelectPrompt}
        disabled
      />,
    );

    const row = screen.getByRole("button", { name: "Summarise my inbox" });
    expect(row).toHaveProperty("disabled", true);

    await userEvent.click(row, { pointerEventsCheck: 0 });

    expect(onSelectPrompt).not.toHaveBeenCalled();
  });
});

describe("OnboardingIntroCard — transcript copy", () => {
  it("offers no copy affordance when there is no transcript", () => {
    render(
      <OnboardingIntroCard
        name="Alex"
        greeting={GREETING}
        prompts={PROMPTS}
        onSelectPrompt={vi.fn()}
      />,
    );

    expect(screen.queryByRole("button", { name: COPY_LABEL })).toBeNull();
  });

  it("copies the transcript verbatim and confirms it visually", async () => {
    const writeText = writeTextMock(() => Promise.resolve());
    const { container } = render(
      <OnboardingIntroCard
        name="Alex"
        greeting={GREETING}
        prompts={PROMPTS}
        transcript="I spend Mondays rebuilding the same report."
        onSelectPrompt={vi.fn()}
      />,
    );

    await userEvent.click(screen.getByRole("button", { name: COPY_LABEL }));

    expect(writeText).toHaveBeenCalledExactlyOnceWith(
      "I spend Mondays rebuilding the same report.",
    );
    await waitFor(() =>
      expect(container.querySelector(".text-emerald-600")).not.toBeNull(),
    );
    expect(toastSpy).not.toHaveBeenCalled();
  });

  it("warns instead of claiming a copy the browser refused", async () => {
    const writeText = writeTextMock(() =>
      Promise.reject(new Error("clipboard blocked")),
    );
    const { container } = render(
      <OnboardingIntroCard
        name="Alex"
        greeting={GREETING}
        prompts={PROMPTS}
        transcript="I spend Mondays rebuilding the same report."
        onSelectPrompt={vi.fn()}
      />,
    );

    await userEvent.click(screen.getByRole("button", { name: COPY_LABEL }));

    expect(writeText).toHaveBeenCalledOnce();
    await waitFor(() =>
      expect(toastSpy).toHaveBeenCalledWith({
        title: "Could not copy the transcript",
        description: "Your browser blocked clipboard access.",
        variant: "destructive",
      }),
    );
    // No tick: the copy never happened.
    expect(container.querySelector(".text-emerald-600")).toBeNull();
  });

  it("drops the confirmation tick again after a couple of seconds", async () => {
    vi.useFakeTimers();
    writeTextMock(() => Promise.resolve());
    const { container } = render(
      <OnboardingIntroCard
        name="Alex"
        greeting={GREETING}
        prompts={PROMPTS}
        transcript="I spend Mondays rebuilding the same report."
        onSelectPrompt={vi.fn()}
      />,
    );

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: COPY_LABEL }));
    });
    expect(container.querySelector(".text-emerald-600")).not.toBeNull();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });

    expect(container.querySelector(".text-emerald-600")).toBeNull();
    vi.useRealTimers();
  });
});

describe("introRevealTimings", () => {
  it("pushes every stage back as the greeting gets longer", () => {
    const short = introRevealTimings("Hello there", 0);
    const long = introRevealTimings("Hello there, this is a longer line", 0);

    expect(short.promptsStart).toBeCloseTo(0.35 + 2 * 0.08 + 0.3, 5);
    expect(long.promptsStart).toBeGreaterThan(short.promptsStart);
    expect(long.footerStart).toBeGreaterThan(short.footerStart);
    expect(long.composerStart).toBeGreaterThan(short.composerStart);
  });

  it("staggers the footer behind the last prompt row and the composer behind the footer", () => {
    const none = introRevealTimings("Hello there", 0);
    const three = introRevealTimings("Hello there", 3);

    expect(three.promptsStart).toBe(none.promptsStart);
    expect(three.footerStart - none.footerStart).toBeCloseTo(3 * 0.12, 5);
    expect(three.composerStart - three.footerStart).toBeCloseTo(0.4, 5);
  });

  it("ignores empty words so padding does not delay the reveal", () => {
    const padded = introRevealTimings("  Hello   there  ", 2);
    const tidy = introRevealTimings("Hello there", 2);

    expect(padded).toEqual(tidy);
  });

  it("still produces an ordered schedule for an empty greeting", () => {
    const { promptsStart, footerStart, composerStart } = introRevealTimings(
      "",
      0,
    );

    expect(promptsStart).toBeCloseTo(0.65, 5);
    expect(footerStart).toBeGreaterThan(promptsStart);
    expect(composerStart).toBeGreaterThan(footerStart);
  });
});
