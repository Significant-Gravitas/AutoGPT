import type { PlanResponse } from "@/app/api/__generated__/models/planResponse";
import type { RecommendedExpert } from "@/app/api/__generated__/models/recommendedExpert";
import type { SuggestedTask } from "@/app/api/__generated__/models/suggestedTask";
import { render, screen, within } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, test } from "vitest";
import { useOnboardingWizardStore } from "../../../../../store";
import { PlanPanel } from "../PlanPanel";

const OTTO_AVATAR = "Otto, your personal Head of AI";

const MARIA: RecommendedExpert = {
  template_id: "tpl-maria",
  name: "Maria",
  role: "SEO & Content",
  avatar_url: null,
};
const MAX: RecommendedExpert = {
  template_id: "tpl-max",
  name: "Max",
  role: "Sales",
  avatar_url: null,
};
const MINA: RecommendedExpert = {
  template_id: "tpl-mina",
  name: "Mina",
  role: "Chief of Staff",
  avatar_url: null,
};

const NEWSLETTER: SuggestedTask = {
  id: "t1",
  title: "Draft this week's newsletter",
  prompt: "Draft this week's newsletter",
  why: "Your list hasn't heard from you in a month.",
  expert_template_id: "tpl-maria",
};
const LEADS: SuggestedTask = {
  id: "t2",
  title: "Follow up with last week's leads",
  prompt: "Follow up with last week's leads",
  why: null,
  expert_template_id: "tpl-max",
};
const INVOICES: SuggestedTask = {
  id: "t3",
  title: "Sort the invoices inbox",
  prompt: "Sort the invoices inbox",
  why: "Three suppliers are waiting on you.",
  expert_template_id: null,
};
const LAUNCH: SuggestedTask = {
  id: "t4",
  title: "Plan the spring launch",
  prompt: "Plan the spring launch",
  expert_template_id: "tpl-mina",
};
const HIRING: SuggestedTask = {
  id: "t5",
  title: "Write the first job post",
  prompt: "Write the first job post",
  expert_template_id: "tpl-mina",
};

function makePlan(overrides: Partial<PlanResponse> = {}): PlanResponse {
  return {
    ready: true,
    source: "llm",
    skipped_dump: false,
    heard: "You run a bakery and lose your evenings to invoices.",
    team: [MARIA, MAX, MINA],
    tasks: [NEWSLETTER, LEADS, INVOICES],
    ...overrides,
  };
}

// The owner line is split across spans; match the element whose whole text
// is the line, not an ancestor that only contains it.
function hasText(text: string) {
  return function match(_content: string, element: Element | null) {
    if (element?.textContent !== text) return false;
    return Array.from(element.children).every(
      (child) => child.textContent !== text,
    );
  };
}

function getTaskCards() {
  const list = screen.getByRole("list", { name: "Your first tasks" });
  return within(list).getAllByRole("listitem");
}

beforeEach(() => {
  useOnboardingWizardStore.getState().reset();
});

describe("PlanPanel header", () => {
  test("introduces Otto's plan with his avatar and a second-level heading", () => {
    render(<PlanPanel plan={makePlan()} moment="trial" />);

    const panel = screen.getByTestId("plan-panel");
    expect(within(panel).getByText("Otto's plan")).toBeDefined();
    expect(
      within(panel).getAllByRole("img", { name: OTTO_AVATAR }).length,
    ).toBeGreaterThan(0);
    expect(
      screen.getByRole("heading", {
        level: 2,
        name: "Here's what your team does first",
      }),
    ).toBeDefined();
    expect(screen.queryByRole("heading", { level: 1 })).toBeNull();
  });

  test("makes the heading the page title when the plan stands alone", () => {
    render(<PlanPanel plan={makePlan()} moment="continue" standalone />);

    expect(
      screen.getByRole("heading", {
        level: 1,
        name: "Here's what your team does first",
      }),
    ).toBeDefined();
    expect(screen.queryByRole("heading", { level: 2 })).toBeNull();
  });
});

describe("PlanPanel heard card", () => {
  test("repeats what Otto heard for a plan built from the brain dump", () => {
    render(<PlanPanel plan={makePlan()} moment="trial" />);

    expect(screen.getByText("What I heard")).toBeDefined();
    expect(
      screen.getByText("You run a bakery and lose your evenings to invoices."),
    ).toBeDefined();
    expect(screen.queryByText("Based on your role")).toBeNull();
  });

  test("speaks to the role for a skipped dump", () => {
    useOnboardingWizardStore.setState({ role: "Engineering" });

    render(
      <PlanPanel
        plan={makePlan({ source: "fallback", skipped_dump: true, heard: null })}
        moment="trial"
      />,
    );

    expect(screen.getByText("Based on your role")).toBeDefined();
    expect(
      screen.getByText(
        "Engineering. You skipped the brain dump, so this is a first plan for an engineer. Tell me about your business once you're in and I'll sharpen it.",
      ),
    ).toBeDefined();
    expect(screen.queryByText("What I heard")).toBeNull();
  });

  test("uses the role's own label and noun", () => {
    useOnboardingWizardStore.setState({ role: "Founder/CEO" });

    render(
      <PlanPanel
        plan={makePlan({ source: "fallback", skipped_dump: true, heard: null })}
        moment="trial"
      />,
    );

    expect(
      screen.getByText(
        "Founder / CEO. You skipped the brain dump, so this is a first plan for a founder. Tell me about your business once you're in and I'll sharpen it.",
      ),
    ).toBeDefined();
  });

  test("keeps the line role-free for Other", () => {
    useOnboardingWizardStore.setState({ role: "Other" });

    render(
      <PlanPanel
        plan={makePlan({ source: "fallback", skipped_dump: true, heard: null })}
        moment="trial"
      />,
    );

    expect(
      screen.getByText(
        "You skipped the brain dump, so this is a first plan to get going. Tell me about your work once you're in and I'll sharpen it.",
      ),
    ).toBeDefined();
  });

  test("does not call a dump that was not skipped a skip", () => {
    useOnboardingWizardStore.setState({ role: "Operations" });

    render(
      <PlanPanel
        plan={makePlan({ source: "fallback", skipped_dump: false })}
        moment="trial"
      />,
    );

    expect(screen.getByText("Based on your role")).toBeDefined();
    expect(
      screen.getByText(
        "Operations. This is a first plan for an operations lead. Tell me about your business once you're in and I'll sharpen it.",
      ),
    ).toBeDefined();
    expect(screen.queryByText(/You skipped the brain dump/)).toBeNull();
  });

  test("never shows heard text on a role plan", () => {
    useOnboardingWizardStore.setState({ role: "Marketing" });

    render(
      <PlanPanel
        plan={makePlan({ source: "fallback", heard: "Leftover dump text." })}
        moment="trial"
      />,
    );

    expect(screen.getByText("Based on your role")).toBeDefined();
    expect(screen.queryByText("Leftover dump text.")).toBeNull();
    expect(screen.queryByText("What I heard")).toBeNull();
  });
});

describe("PlanPanel task cards", () => {
  test("shows the first three tasks, in order", () => {
    render(
      <PlanPanel
        plan={makePlan({
          tasks: [NEWSLETTER, LEADS, INVOICES, LAUNCH, HIRING],
        })}
        moment="trial"
      />,
    );

    const cards = getTaskCards();
    expect(cards).toHaveLength(3);
    expect(within(cards[0]).getByText(NEWSLETTER.title)).toBeDefined();
    expect(within(cards[1]).getByText(LEADS.title)).toBeDefined();
    expect(within(cards[2]).getByText(INVOICES.title)).toBeDefined();
    expect(screen.queryByText(LAUNCH.title)).toBeNull();
    expect(screen.queryByText(HIRING.title)).toBeNull();
  });

  test("shows fewer cards when the plan has fewer tasks", () => {
    render(<PlanPanel plan={makePlan({ tasks: [LEADS] })} moment="trial" />);

    const cards = getTaskCards();
    expect(cards).toHaveLength(1);
    expect(within(cards[0]).getByText(LEADS.title)).toBeDefined();
  });

  test("marks only the first card to start with", () => {
    render(<PlanPanel plan={makePlan()} moment="trial" />);

    const cards = getTaskCards();
    expect(within(cards[0]).getByText("Start here")).toBeDefined();
    expect(within(cards[1]).queryByText("Start here")).toBeNull();
    expect(within(cards[2]).queryByText("Start here")).toBeNull();
  });

  test("gives each card its owner's face, title, reason and owner line", () => {
    render(<PlanPanel plan={makePlan()} moment="trial" />);

    const [newsletter] = getTaskCards();
    expect(
      within(newsletter).getByRole("img", { name: "Maria, AI Expert" }),
    ).toBeDefined();
    expect(within(newsletter).getByText(NEWSLETTER.title)).toBeDefined();
    expect(within(newsletter).getByText(NEWSLETTER.why!)).toBeDefined();
    expect(
      within(newsletter).getByText(
        hasText("Maria, SEO Content Manager · joins your team"),
      ),
    ).toBeDefined();
  });

  test("leaves the reason out when the task has none", () => {
    render(<PlanPanel plan={makePlan()} moment="trial" />);

    const leads = getTaskCards()[1];
    // Title and owner line only: no third line of text in between.
    expect(leads.textContent).toBe(
      `${LEADS.title}Max, Sales Development Rep · joins your team`,
    );
  });

  test("shows a role with no display label as it is", () => {
    render(<PlanPanel plan={makePlan({ tasks: [LAUNCH] })} moment="trial" />);

    const [launch] = getTaskCards();
    expect(
      within(launch).getByText(
        hasText("Mina, Chief of Staff · joins your team"),
      ),
    ).toBeDefined();
  });

  test("gives Otto a task with no owner", () => {
    render(<PlanPanel plan={makePlan()} moment="trial" />);

    const invoices = getTaskCards()[2];
    expect(within(invoices).getByText("Otto handles this one")).toBeDefined();
    expect(
      within(invoices).getByRole("img", { name: OTTO_AVATAR }),
    ).toBeDefined();
    expect(within(invoices).queryByText(/joins your team/)).toBeNull();
  });

  test("gives Otto a task whose owner is not on the team", () => {
    render(
      <PlanPanel
        plan={makePlan({ team: [MARIA], tasks: [NEWSLETTER, LEADS] })}
        moment="trial"
      />,
    );

    const leads = getTaskCards()[1];
    expect(within(leads).getByText("Otto handles this one")).toBeDefined();
    expect(within(leads).getByRole("img", { name: OTTO_AVATAR })).toBeDefined();
    expect(
      within(leads).queryByRole("img", { name: "Max, AI Expert" }),
    ).toBeNull();
  });

  test("offers nothing to click or focus", async () => {
    const user = userEvent.setup();
    render(<PlanPanel plan={makePlan()} moment="trial" />);

    expect(screen.queryAllByRole("button")).toHaveLength(0);
    expect(screen.queryAllByRole("link")).toHaveLength(0);
    await user.tab();
    expect(document.activeElement).toBe(document.body);
  });
});

describe("PlanPanel consent line", () => {
  test("names every expert who joins when the trial starts", () => {
    render(<PlanPanel plan={makePlan()} moment="trial" />);

    expect(
      screen.getByText(
        "Maria, Max and Mina join your team when your trial starts. Nothing runs until you pick a task.",
      ),
    ).toBeDefined();
  });

  test("uses the singular for one expert", () => {
    render(<PlanPanel plan={makePlan({ team: [MARIA] })} moment="trial" />);

    expect(
      screen.getByText(
        "Maria joins your team when your trial starts. Nothing runs until you pick a task.",
      ),
    ).toBeDefined();
  });

  test("leaves the tasks with Otto when nobody joins", () => {
    render(
      <PlanPanel
        plan={makePlan({ team: [], tasks: [INVOICES] })}
        moment="trial"
      />,
    );

    expect(
      screen.getByText(
        "Otto takes these on for now. You can hire experts any time. Nothing runs until you pick a task.",
      ),
    ).toBeDefined();
  });

  test("names the moment the team joins", () => {
    const { unmount } = render(
      <PlanPanel plan={makePlan()} moment="subscribe" />,
    );
    expect(
      screen.getByText(
        "Maria, Max and Mina join your team when you subscribe. Nothing runs until you pick a task.",
      ),
    ).toBeDefined();
    unmount();

    render(<PlanPanel plan={makePlan()} moment="continue" />);
    expect(
      screen.getByText(
        "Maria, Max and Mina join your team when you continue. Nothing runs until you pick a task.",
      ),
    ).toBeDefined();
  });
});

describe("PlanPanel while the plan is being written", () => {
  test("shows skeletons and says Otto is writing", () => {
    render(<PlanPanel plan={null} moment="trial" />);

    const panel = screen.getByTestId("plan-panel");
    expect(within(panel).getByTestId("plan-panel-pending")).toBeDefined();
    expect(within(panel).getByRole("status").textContent).toBe(
      "Otto is writing your plan",
    );
    expect(
      screen.getByRole("heading", { name: "Here's what your team does first" }),
    ).toBeDefined();
    expect(screen.queryByRole("list", { name: "Your first tasks" })).toBeNull();
    expect(screen.queryByText(/Nothing runs until you pick a task/)).toBeNull();
  });

  test("drops the skeletons and announces the plan once it arrives", () => {
    const { rerender } = render(<PlanPanel plan={null} moment="trial" />);
    const status = screen.getByRole("status");

    rerender(<PlanPanel plan={makePlan()} moment="trial" />);

    expect(screen.queryByTestId("plan-panel-pending")).toBeNull();
    // The same live region, outside the busy skeleton, so the change is read.
    expect(screen.getByRole("status")).toBe(status);
    expect(status.textContent).toBe("Your plan is ready");
    expect(status.closest("[aria-busy]")).toBeNull();
  });

  test("starts with the tasks and the team both being worked out", () => {
    render(<PlanPanel plan={null} moment="trial" />);

    expect(progressSteps()).toEqual([
      ["Turning your brain dump into tasks", "In progress"],
      ["Finding experts who fit your work", "In progress"],
      ["Choosing who takes each task", "Pending"],
    ]);
  });

  test("ticks off each step the server reports done", () => {
    const { rerender } = render(
      <PlanPanel
        plan={null}
        progress={{ tasks_written: false, team_chosen: true }}
        moment="trial"
      />,
    );
    expect(progressSteps()).toEqual([
      ["Turning your brain dump into tasks", "In progress"],
      ["Finding experts who fit your work", "Completed"],
      ["Choosing who takes each task", "Pending"],
    ]);

    rerender(
      <PlanPanel
        plan={null}
        progress={{ tasks_written: true, team_chosen: true }}
        moment="trial"
      />,
    );
    expect(progressSteps()).toEqual([
      ["Turning your brain dump into tasks", "Completed"],
      ["Finding experts who fit your work", "Completed"],
      ["Choosing who takes each task", "In progress"],
    ]);
  });
});

// Each step's label and the state its mark reads out.
function progressSteps() {
  const list = screen.getByRole("list", { name: "Writing your plan" });
  return within(list)
    .getAllByRole("listitem")
    .map((item) => {
      const mark = item.querySelector(".sr-only")?.textContent ?? "";
      return [item.textContent?.replace(mark, "") ?? "", mark];
    });
}
