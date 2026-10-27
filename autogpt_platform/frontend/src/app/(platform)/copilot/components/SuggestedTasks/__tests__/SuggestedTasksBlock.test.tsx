import type { SuggestedTask } from "@/app/api/__generated__/models/suggestedTask";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { type ComponentProps, useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ExpertIdentity } from "../../../useExpertMap";
import { SuggestedTasksBlock } from "../SuggestedTasksBlock";
import { useCollapsedSuggestions } from "../useCollapsedSuggestions";

const motionPreference = vi.hoisted(() => ({ reduced: false }));
vi.mock("framer-motion", async (importOriginal) => ({
  ...(await importOriginal<typeof import("framer-motion")>()),
  useReducedMotion: () => motionPreference.reduced,
}));

const capture = vi.hoisted(() => vi.fn());
vi.mock("posthog-js", () => ({ default: { capture } }));

const auth = vi.hoisted(() => ({ userID: "user-1" }));
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: auth.userID },
    isUserLoading: false,
    isLoggedIn: true,
  }),
}));

type Suggestions = ComponentProps<typeof SuggestedTasksBlock>["suggestions"];

const MARIA: ExpertIdentity = {
  id: "expert-maria",
  name: "Maria",
  avatarUrl: null,
  role: "Finance, Invoicing & Bookkeeping",
  isArchived: false,
  readOnlyReason: null,
};

const SAM: ExpertIdentity = {
  id: "expert-sam",
  name: "Sam",
  avatarUrl: null,
  role: null,
  isArchived: false,
  readOnlyReason: null,
};

// Four rows the user can see (three open, one started) and one removed, so
// the folded count and the open count differ.
const TASKS: SuggestedTask[] = [
  {
    id: "t0",
    title: "Chase unpaid invoices",
    prompt: "Chase every unpaid invoice with a polite follow-up.",
    why: "Three invoices are past due.",
    state: "open",
    expert_id: MARIA.id,
  },
  {
    id: "t1",
    title: "Draft the launch post",
    prompt: "Draft a launch post for the new plan.",
    state: "sent",
    session_id: "chat-1",
  },
  {
    id: "t2",
    title: "Tidy the shared drive",
    prompt: "Sort the shared drive into one folder per client.",
    state: "dismissed",
  },
  {
    id: "t3",
    title: "Watch competitor pricing",
    prompt: "Check competitor pricing weekly and tell me what moved.",
    state: "open",
  },
  {
    id: "t4",
    title: "Book the team offsite",
    prompt: "Find three venues for a two-day offsite in May.",
    state: "open",
    expert_id: SAM.id,
  },
];

const FIRST_TASKS = "Your first tasks";
const COLLAPSED_KEY = "home.suggestedTasks.collapsed";

function makeSuggestions(
  allTasks: SuggestedTask[] = TASKS,
  overrides: Partial<Suggestions> = {},
): Suggestions {
  return {
    isStored: true,
    allTasks,
    tasks: allTasks.filter((task) => task.state !== "dismissed"),
    experts: [MARIA, SAM],
    expertsById: new Map([MARIA, SAM].map((expert) => [expert.id, expert])),
    isRosterLoaded: true,
    sendingId: null,
    sendTask: vi.fn(),
    removeTask: vi.fn(),
    ...overrides,
  };
}

interface HarnessProps {
  suggestions?: Suggestions;
  initiallyCollapsed?: boolean;
  onToggle?: (openRows: number) => void;
  showTeam?: boolean;
}

// Holds the collapsed state the way the page does, and reports each toggle.
function Harness({
  suggestions = makeSuggestions(),
  initiallyCollapsed = false,
  onToggle = () => {},
  showTeam,
}: HarnessProps) {
  const [isCollapsed, setIsCollapsed] = useState(initiallyCollapsed);
  function toggle(openRows: number) {
    onToggle(openRows);
    setIsCollapsed(!isCollapsed);
  }
  return (
    <SuggestedTasksBlock
      suggestions={suggestions}
      collapse={{ isCollapsed, toggle }}
      showTeam={showTeam}
    />
  );
}

// The block with the stored per-user preference behind it.
function StoredHarness() {
  const collapse = useCollapsedSuggestions();
  return (
    <SuggestedTasksBlock suggestions={makeSuggestions()} collapse={collapse} />
  );
}

function controlledRegion(toggle: HTMLElement) {
  const id = toggle.getAttribute("aria-controls");
  expect(id).toBeTruthy();
  return document.getElementById(id ?? "");
}

function events(name: string) {
  return capture.mock.calls
    .filter(([event]) => event === name)
    .map(([, properties]) => properties);
}

beforeEach(() => {
  capture.mockReset();
  window.localStorage.clear();
  auth.userID = "user-1";
});

afterEach(() => {
  motionPreference.reduced = false;
});

describe("SuggestedTasksBlock — Hide and Show", () => {
  it("shows the tasks under their header with a Hide toggle", () => {
    render(<Harness />);

    const block = screen.getByRole("region", {
      name: "Suggested first tasks",
    });
    expect(within(block).getByText("Suggested first tasks")).toBeDefined();
    const toggle = screen.getByRole("button", { name: "Hide" });
    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    const list = screen.getByRole("list", { name: FIRST_TASKS });
    expect(controlledRegion(toggle)?.contains(list)).toBe(true);
    expect(
      screen.getByRole("button", {
        name: `Start with Maria: ${TASKS[0].title}`,
      }),
    ).toBeDefined();
  });

  it("folds the list away on Hide and says how many tasks are behind Show", async () => {
    const onToggle = vi.fn();
    render(<Harness onToggle={onToggle} />);

    await userEvent.click(screen.getByRole("button", { name: "Hide" }));

    const toggle = await screen.findByRole("button", {
      name: "4 tasks · Show",
    });
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    await waitFor(() =>
      expect(screen.queryByRole("list", { name: FIRST_TASKS })).toBeNull(),
    );
    expect(screen.queryByText(TASKS[0].title)).toBeNull();
    expect(onToggle).toHaveBeenCalledExactlyOnceWith(3);
  });

  it("brings the list back on Show", async () => {
    const onToggle = vi.fn();
    render(<Harness initiallyCollapsed onToggle={onToggle} />);

    expect(screen.queryByRole("list", { name: FIRST_TASKS })).toBeNull();
    await userEvent.click(
      screen.getByRole("button", { name: "4 tasks · Show" }),
    );

    const toggle = await screen.findByRole("button", { name: "Hide" });
    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    expect(
      await screen.findByRole("list", { name: FIRST_TASKS }),
    ).toBeDefined();
    expect(screen.getByText(TASKS[0].title)).toBeDefined();
    expect(onToggle).toHaveBeenCalledExactlyOnceWith(3);
  });

  it("points the toggle at the same region whether the list is open or folded", async () => {
    render(<Harness />);

    const open = screen.getByRole("button", { name: "Hide" });
    const region = controlledRegion(open);
    expect(region).not.toBeNull();

    await userEvent.click(open);

    const folded = await screen.findByRole("button", {
      name: "4 tasks · Show",
    });
    expect(folded.getAttribute("aria-controls")).toBe(
      open.getAttribute("aria-controls"),
    );
    expect(controlledRegion(folded)).toBe(region);
    await waitFor(() =>
      expect(screen.queryByRole("list", { name: FIRST_TASKS })).toBeNull(),
    );
    expect(controlledRegion(folded)).not.toBeNull();
  });

  it("says one task when only one is left to show", () => {
    render(
      <Harness
        initiallyCollapsed
        suggestions={makeSuggestions([TASKS[2], TASKS[3]])}
      />,
    );

    expect(screen.getByRole("button", { name: "1 task · Show" })).toBeDefined();
  });

  it("counts started tasks behind Show but reports only the open ones when toggled", async () => {
    const onToggle = vi.fn();
    render(
      <Harness
        initiallyCollapsed
        onToggle={onToggle}
        suggestions={makeSuggestions([TASKS[1], TASKS[2]])}
      />,
    );

    await userEvent.click(
      screen.getByRole("button", { name: "1 task · Show" }),
    );

    expect(await screen.findByRole("button", { name: "Hide" })).toBeDefined();
    expect(onToggle).toHaveBeenCalledExactlyOnceWith(0);
  });

  it("starts and removes tasks through the suggestions it is given", async () => {
    const suggestions = makeSuggestions();
    render(<Harness suggestions={suggestions} />);

    await userEvent.click(
      screen.getByRole("button", {
        name: `Start with Maria: ${TASKS[0].title}`,
      }),
    );
    await userEvent.click(
      screen.getByRole("button", { name: `Remove: ${TASKS[3].title}` }),
    );

    expect(suggestions.sendTask).toHaveBeenCalledExactlyOnceWith(TASKS[0]);
    expect(suggestions.removeTask).toHaveBeenCalledExactlyOnceWith(TASKS[3]);
  });
});

describe("SuggestedTasksBlock — the stored preference", () => {
  it("keeps the list folded for the same user after a reload and reports each change", async () => {
    const first = render(<StoredHarness />);

    await userEvent.click(screen.getByRole("button", { name: "Hide" }));

    expect(
      await screen.findByRole("button", { name: "4 tasks · Show" }),
    ).toBeDefined();
    expect(events("home_suggestions_collapsed")).toEqual([{ open: 3 }]);
    expect(
      JSON.parse(window.localStorage.getItem(COLLAPSED_KEY) ?? ""),
    ).toEqual(["user-1"]);
    first.unmount();

    render(<StoredHarness />);

    expect(screen.queryByRole("list", { name: FIRST_TASKS })).toBeNull();
    await userEvent.click(
      screen.getByRole("button", { name: "4 tasks · Show" }),
    );

    expect(
      await screen.findByRole("list", { name: FIRST_TASKS }),
    ).toBeDefined();
    expect(events("home_suggestions_expanded")).toEqual([{ open: 3 }]);
    expect(
      JSON.parse(window.localStorage.getItem(COLLAPSED_KEY) ?? ""),
    ).toEqual([]);
  });

  it("shows the list to another user on the same browser", () => {
    window.localStorage.setItem(COLLAPSED_KEY, JSON.stringify(["user-1"]));
    auth.userID = "user-2";

    render(<StoredHarness />);

    expect(screen.getByRole("button", { name: "Hide" })).toBeDefined();
    expect(screen.getByRole("list", { name: FIRST_TASKS })).toBeDefined();
  });
});

describe("SuggestedTasksBlock — Your team", () => {
  it("lists Otto and each expert on the team with their role", () => {
    render(<Harness showTeam />);

    const team = screen.getByRole("list", { name: "Your team" });
    const members = within(team).getAllByRole("listitem");
    expect(members).toHaveLength(3);

    expect(within(members[0]).getByText("Otto")).toBeDefined();
    expect(within(members[0]).getByText("Head of AI")).toBeDefined();
    expect(within(members[1]).getByText("Maria")).toBeDefined();
    expect(within(members[1]).getByText("Bookkeeper")).toBeDefined();
    expect(within(members[2]).getByText("Sam")).toBeDefined();
  });

  it("lists only Otto when no expert has been hired", () => {
    render(
      <Harness
        showTeam
        suggestions={makeSuggestions(TASKS, { experts: [] })}
      />,
    );

    const team = screen.getByRole("list", { name: "Your team" });
    const members = within(team).getAllByRole("listitem");
    expect(members).toHaveLength(1);
    expect(within(members[0]).getByText("Otto")).toBeDefined();
  });

  it("leaves the team out unless asked to show it", () => {
    render(<Harness />);

    expect(screen.getByRole("list", { name: FIRST_TASKS })).toBeDefined();
    expect(screen.queryByRole("list", { name: "Your team" })).toBeNull();
  });

  it("folds the team away with the list", async () => {
    render(<Harness showTeam />);

    await userEvent.click(screen.getByRole("button", { name: "Hide" }));

    await waitFor(() =>
      expect(screen.queryByRole("list", { name: "Your team" })).toBeNull(),
    );
  });
});

describe("SuggestedTasksBlock — nothing to suggest", () => {
  it("renders nothing when there are no tasks", () => {
    render(<Harness showTeam suggestions={makeSuggestions([])} />);

    expect(
      screen.queryByRole("region", { name: "Suggested first tasks" }),
    ).toBeNull();
    expect(screen.queryByRole("button")).toBeNull();
    expect(screen.queryByRole("list")).toBeNull();
  });

  it("renders nothing once every task has been removed", () => {
    render(
      <Harness
        showTeam
        suggestions={makeSuggestions([
          { ...TASKS[0], state: "dismissed" },
          TASKS[2],
        ])}
      />,
    );

    expect(
      screen.queryByRole("region", { name: "Suggested first tasks" }),
    ).toBeNull();
    expect(screen.queryByText("Suggested first tasks")).toBeNull();
  });
});

describe("SuggestedTasksBlock — reduced motion", () => {
  it("still hides and shows the list when the user prefers less motion", async () => {
    motionPreference.reduced = true;
    render(<Harness />);

    await userEvent.click(screen.getByRole("button", { name: "Hide" }));

    expect(
      await screen.findByRole("button", { name: "4 tasks · Show" }),
    ).toBeDefined();
    await waitFor(() =>
      expect(screen.queryByRole("list", { name: FIRST_TASKS })).toBeNull(),
    );

    await userEvent.click(
      screen.getByRole("button", { name: "4 tasks · Show" }),
    );

    expect(
      await screen.findByRole("list", { name: FIRST_TASKS }),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Hide" })).toBeDefined();
  });
});

it("shows nothing until the roster has answered, so no owner reads as Otto", () => {
  render(
    <SuggestedTasksBlock
      suggestions={makeSuggestions(TASKS, { isRosterLoaded: false })}
      collapse={{ isCollapsed: false, toggle: vi.fn() }}
    />,
  );

  expect(
    screen.queryByRole("region", { name: "Suggested first tasks" }),
  ).toBeNull();
});
