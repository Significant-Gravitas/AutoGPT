import type { SuggestedTask } from "@/app/api/__generated__/models/suggestedTask";
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

const PROMPTS: SuggestedTask[] = [
  {
    id: "0",
    title: "Summarise my inbox",
    prompt: "Every morning, summarise my unread email and flag what matters.",
    icon: "envelope",
  },
  {
    id: "1",
    title: "Watch competitor pricing",
    prompt: "Check competitor pricing pages weekly and tell me what moved.",
    icon: "not-a-real-icon-slug",
  },
];

const GREETING = "You spend most of your week chasing updates.";
const COPY_LABEL = "Copy your recording's transcript";
const FOOTER = "Want to do something else? Just write it in the textbox below.";
const EMPTY_FOOTER =
  "Tell me what you'd like to start with in the textbox below.";

function makeSuggestions(
  tasks: SuggestedTask[],
  sendTask: (task: SuggestedTask) => void = vi.fn(),
) {
  return {
    isStored: true,
    tasks,
    sendingId: null,
    sendTask: async (task: SuggestedTask) => sendTask(task),
    hideTask: vi.fn(),
    hideAll: vi.fn(),
  };
}

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
        suggestions={makeSuggestions(PROMPTS)}
      />,
    );

    const text = normalizeWhitespace(container);
    expect(text).toContain("Hey, Alex");
    expect(text).toContain(GREETING);

    expect(screen.getByText("Summarise my inbox")).toBeDefined();
    expect(screen.getByText("Watch competitor pricing")).toBeDefined();
    expect(screen.getByText(FOOTER)).toBeDefined();
  });

  it("says Hey there to a user it has no name for", () => {
    const { container } = render(
      <OnboardingIntroCard
        name={null}
        greeting={GREETING}
        suggestions={makeSuggestions(PROMPTS)}
      />,
    );

    expect(normalizeWhitespace(container)).toContain("Hey there");
  });

  it("renders a row for a prompt whose icon slug the frontend does not know", () => {
    render(
      <OnboardingIntroCard
        name="Alex"
        greeting={GREETING}
        suggestions={makeSuggestions([PROMPTS[1]])}
      />,
    );

    expect(
      screen.getByRole("button", { name: "Watch competitor pricing" }),
    ).toBeDefined();
  });

  it("omits the list and stops pointing at it once nothing is left", () => {
    render(
      <OnboardingIntroCard
        name="Alex"
        greeting={GREETING}
        suggestions={makeSuggestions([])}
      />,
    );

    expect(screen.queryByRole("list")).toBeNull();
    expect(screen.getByText(EMPTY_FOOTER)).toBeDefined();
    expect(screen.queryByText(FOOTER)).toBeNull();
  });
});

describe("OnboardingIntroCard — picking a prompt", () => {
  it("hands the clicked row's task over to be sent", async () => {
    const onSelectPrompt = vi.fn();
    render(
      <OnboardingIntroCard
        name="Alex"
        greeting={GREETING}
        suggestions={makeSuggestions(PROMPTS, onSelectPrompt)}
      />,
    );

    await userEvent.click(
      screen.getByRole("button", { name: "Summarise my inbox" }),
    );

    expect(onSelectPrompt).toHaveBeenCalledExactlyOnceWith(PROMPTS[0]);
  });

  it("does not fire while the composer is disabled", async () => {
    const onSelectPrompt = vi.fn();
    render(
      <OnboardingIntroCard
        name="Alex"
        greeting={GREETING}
        suggestions={makeSuggestions(PROMPTS, onSelectPrompt)}
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
        suggestions={makeSuggestions(PROMPTS)}
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
        suggestions={makeSuggestions(PROMPTS)}
        transcript="I spend Mondays rebuilding the same report."
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
        suggestions={makeSuggestions(PROMPTS)}
        transcript="I spend Mondays rebuilding the same report."
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
        suggestions={makeSuggestions(PROMPTS)}
        transcript="I spend Mondays rebuilding the same report."
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
  it("orders greeting, prompts, footer and composer with fixed offsets", () => {
    const { promptsStart, footerStart, composerStart } = introRevealTimings();

    expect(promptsStart).toBeCloseTo(0.5 + 0.45 + 0.3, 5);
    expect(footerStart - promptsStart).toBeCloseTo(0.35, 5);
    expect(composerStart - footerStart).toBeCloseTo(0.4, 5);
  });
});
