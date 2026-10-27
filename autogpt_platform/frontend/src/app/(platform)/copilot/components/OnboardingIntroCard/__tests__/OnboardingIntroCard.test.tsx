import type { SuggestedTask } from "@/app/api/__generated__/models/suggestedTask";
import {
  act,
  cleanup,
  fireEvent,
  normalizeWhitespace,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import type { ComponentProps } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ExpertIdentity } from "../../../useExpertMap";
import {
  introRevealTimings,
  OnboardingIntroCard,
} from "../OnboardingIntroCard";

const toastSpy = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: (...args: unknown[]) => toastSpy(...(args as [])),
  useToast: () => ({ toast: toastSpy }),
}));

type CardProps = ComponentProps<typeof OnboardingIntroCard>;
type Suggestions = CardProps["suggestions"];
type Collapse = CardProps["collapse"];

const MARIA: ExpertIdentity = {
  id: "expert-maria",
  name: "Maria",
  avatarUrl: null,
  role: "Finance, Invoicing & Bookkeeping",
  isArchived: false,
  readOnlyReason: null,
};

const PROMPTS: SuggestedTask[] = [
  {
    id: "0",
    title: "Summarise my inbox",
    prompt: "Every morning, summarise my unread email and flag what matters.",
    icon: "envelope",
    state: "open",
  },
  {
    id: "1",
    title: "Watch competitor pricing",
    prompt: "Check competitor pricing pages weekly and tell me what moved.",
    icon: "not-a-real-icon-slug",
    state: "open",
  },
];

const MARIAS_TASK: SuggestedTask = {
  id: "2",
  title: "Chase unpaid invoices",
  prompt: "Chase every unpaid invoice with a polite follow-up.",
  why: "Three invoices are past due.",
  state: "open",
  expert_id: MARIA.id,
};

const GREETING = "You spend most of your week chasing updates.";
const COPY_LABEL = "Copy your recording's transcript";
const FOOTER = "Want to do something else? Just write it in the textbox below.";
const EMPTY_FOOTER =
  "Tell me what you'd like to start with in the textbox below.";
const TRANSCRIPT = "I spend Mondays rebuilding the same report.";

function makeSuggestions(
  tasks: SuggestedTask[],
  overrides: Partial<Suggestions> = {},
): Suggestions {
  return {
    isStored: true,
    allTasks: tasks,
    tasks: tasks.filter((task) => task.state !== "dismissed"),
    experts: [MARIA],
    expertsById: new Map([[MARIA.id, MARIA]]),
    isRosterLoaded: true,
    sendingId: null,
    sendTask: vi.fn(),
    removeTask: vi.fn(),
    ...overrides,
  };
}

function makeCollapse(isCollapsed = false): Collapse {
  return { isCollapsed, toggle: vi.fn() };
}

function renderCard(props: Partial<CardProps> = {}) {
  return render(
    <OnboardingIntroCard
      name="Alex"
      greeting={GREETING}
      suggestions={makeSuggestions(PROMPTS)}
      collapse={makeCollapse()}
      {...props}
    />,
  );
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
  vi.useRealTimers();
});

describe("OnboardingIntroCard — content", () => {
  it("greets the user by name and renders the greeting, the task list and the footer", () => {
    const { container } = renderCard();

    const text = normalizeWhitespace(container);
    expect(text).toContain("Hey, Alex");
    expect(text).toContain(GREETING);

    const list = screen.getByRole("list", { name: "Your first tasks" });
    expect(
      within(list).getByRole("button", {
        name: "Start with Otto: Summarise my inbox",
      }),
    ).toBeDefined();
    expect(
      within(list).getByRole("button", {
        name: "Start with Otto: Watch competitor pricing",
      }),
    ).toBeDefined();
    expect(screen.getByText(FOOTER)).toBeDefined();
  });

  it("says Hey there to a user it has no name for", () => {
    const { container } = renderCard({ name: null });

    expect(normalizeWhitespace(container)).toContain("Hey there");
  });

  it("names the expert who takes a task and gives the rest to Otto", () => {
    renderCard({ suggestions: makeSuggestions([MARIAS_TASK, PROMPTS[1]]) });

    expect(
      screen.getByRole("button", {
        name: "Start with Maria: Chase unpaid invoices",
      }),
    ).toBeDefined();
    expect(screen.getByText(MARIAS_TASK.why ?? "")).toBeDefined();
    expect(
      screen.getByRole("button", {
        name: "Start with Otto: Watch competitor pricing",
      }),
    ).toBeDefined();
  });

  it("shows who is on the team above the tasks", () => {
    renderCard();

    const team = screen.getByRole("list", { name: "Your team" });
    const members = within(team).getAllByRole("listitem");
    expect(members).toHaveLength(2);
    expect(within(members[0]).getByText("Otto")).toBeDefined();
    expect(within(members[1]).getByText("Maria")).toBeDefined();
    expect(within(members[1]).getByText("Bookkeeper")).toBeDefined();
  });

  it("omits the list and stops pointing at it once nothing is left", () => {
    renderCard({ suggestions: makeSuggestions([]) });

    expect(screen.queryByRole("list")).toBeNull();
    expect(screen.getByText(EMPTY_FOOTER)).toBeDefined();
    expect(screen.queryByText(FOOTER)).toBeNull();
  });
});

describe("OnboardingIntroCard — picking a task", () => {
  it("hands the clicked row's task over to be sent", async () => {
    const suggestions = makeSuggestions(PROMPTS);
    renderCard({ suggestions });

    await userEvent.click(
      screen.getByRole("button", {
        name: "Start with Otto: Summarise my inbox",
      }),
    );

    expect(suggestions.sendTask).toHaveBeenCalledExactlyOnceWith(PROMPTS[0]);
  });

  it("does not fire while the composer is disabled", async () => {
    const suggestions = makeSuggestions(PROMPTS);
    renderCard({ suggestions, disabled: true });

    const row = screen.getByRole("button", {
      name: "Start with Otto: Summarise my inbox",
    });
    expect(row).toHaveProperty("disabled", true);

    await userEvent.click(row, { pointerEventsCheck: 0 });

    expect(suggestions.sendTask).not.toHaveBeenCalled();
  });
});

describe("OnboardingIntroCard — folding the list", () => {
  it("hides the something-else line while the list is folded", () => {
    renderCard({ collapse: makeCollapse(true) });

    expect(
      screen.getByRole("button", { name: "2 tasks · Show" }),
    ).toBeDefined();
    expect(screen.queryByRole("list", { name: "Your first tasks" })).toBeNull();
    expect(screen.queryByText(FOOTER)).toBeNull();
    expect(screen.queryByText(EMPTY_FOOTER)).toBeNull();
  });

  it("shows the something-else line again once the list is open", () => {
    const { rerender } = renderCard({ collapse: makeCollapse(true) });
    expect(screen.queryByText(FOOTER)).toBeNull();

    rerender(
      <OnboardingIntroCard
        name="Alex"
        greeting={GREETING}
        suggestions={makeSuggestions(PROMPTS)}
        collapse={makeCollapse(false)}
      />,
    );

    expect(screen.getByText(FOOTER)).toBeDefined();
    expect(
      screen.getByRole("list", { name: "Your first tasks" }),
    ).toBeDefined();
  });

  it("asks to fold the list when Hide is clicked", async () => {
    const collapse = makeCollapse(false);
    renderCard({ collapse });

    await userEvent.click(screen.getByRole("button", { name: "Hide" }));

    expect(collapse.toggle).toHaveBeenCalledExactlyOnceWith(2);
  });

  it("still says where to start when there are no tasks, even with the list folded", () => {
    renderCard({
      suggestions: makeSuggestions([]),
      collapse: makeCollapse(true),
    });

    expect(screen.getByText(EMPTY_FOOTER)).toBeDefined();
    expect(
      screen.queryByRole("region", { name: "Suggested first tasks" }),
    ).toBeNull();
  });
});

describe("OnboardingIntroCard — transcript copy", () => {
  it("offers no copy affordance when there is no transcript", () => {
    renderCard();

    expect(screen.queryByRole("button", { name: COPY_LABEL })).toBeNull();
  });

  it("copies the transcript verbatim and confirms it visually", async () => {
    const writeText = writeTextMock(() => Promise.resolve());
    const { container } = renderCard({ transcript: TRANSCRIPT });

    await userEvent.click(screen.getByRole("button", { name: COPY_LABEL }));

    expect(writeText).toHaveBeenCalledExactlyOnceWith(TRANSCRIPT);
    await waitFor(() =>
      expect(container.querySelector(".text-emerald-600")).not.toBeNull(),
    );
    expect(toastSpy).not.toHaveBeenCalled();
  });

  it("warns instead of claiming a copy the browser refused", async () => {
    const writeText = writeTextMock(() =>
      Promise.reject(new Error("clipboard blocked")),
    );
    const { container } = renderCard({ transcript: TRANSCRIPT });

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
    const { container } = renderCard({ transcript: TRANSCRIPT });

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
