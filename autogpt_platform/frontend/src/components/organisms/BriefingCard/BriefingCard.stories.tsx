import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { expect, userEvent, within } from "storybook/test";
import type { BriefingResponse } from "@/app/api/__generated__/models/briefingResponse";
import type { BriefingRunItem } from "@/app/api/__generated__/models/briefingRunItem";
import { MANAGED_IDENTITIES } from "@/components/molecules/ExpertAvatar/helpers";
import { BriefingCard } from "./BriefingCard";

const BRIEFING_DATE = new Date("2026-08-12T12:00:00Z");

const RUNS: BriefingRunItem[] = [
  runItem(0, {
    expert_name: "Maria",
    expert_avatar_url: MANAGED_IDENTITIES[0].url,
    agent_name: "Lead finder",
    summary: "Found 3 new leads in fintech, all with a named buyer.",
  }),
  runItem(1, {
    expert_name: "Jules",
    expert_avatar_url: MANAGED_IDENTITIES[1].url,
    agent_name: "Invoice chaser",
    summary: "Sent 4 reminders; 2 invoices are now more than 30 days late.",
  }),
  runItem(2, {
    expert_name: "Nadia",
    expert_avatar_url: MANAGED_IDENTITIES[2].url,
    agent_name: "Morning digest",
    summary: "Summarised 12 newsletters into a five-minute read.",
  }),
  runItem(3, {
    expert_name: "Remy",
    expert_avatar_url: MANAGED_IDENTITIES[3].url,
    agent_name: "Competitor watch",
    summary: "Two competitors changed their pricing pages overnight.",
  }),
  runItem(4, {
    agent_name: "Support triage",
    summary: "Tagged 18 tickets and escalated one billing dispute.",
  }),
  runItem(5, {
    agent_name: "Social scheduler",
    summary: "Queued 5 posts for this week.",
  }),
  runItem(6, {
    agent_name: "Expense sorter",
    summary: "Filed 22 receipts under the right cost centres.",
  }),
  runItem(7, {
    agent_name: "Hiring pipeline",
    summary: "Moved 3 candidates to the interview stage.",
  }),
];

function runItem(
  index: number,
  overrides: Partial<BriefingRunItem> = {},
): BriefingRunItem {
  return {
    expert_id: `exp-${index}`,
    expert_name: "Maria",
    expert_avatar_url: null,
    agent_name: `Agent ${index}`,
    graph_id: `g-${index}`,
    execution_id: `run-${index}`,
    library_agent_id: `lib-${index}`,
    status: "COMPLETED",
    summary: `Summary ${index}`,
    link: `/library/agents/lib-${index}`,
    ...overrides,
  };
}

function briefing(items: BriefingRunItem[]): BriefingResponse {
  return {
    id: "briefing-1",
    briefing_date: BRIEFING_DATE,
    created_at: BRIEFING_DATE,
    delivered_at: null,
    content: {
      generated_at: BRIEFING_DATE,
      timezone: "UTC",
      zero_expert_fallback: false,
      run_items: items,
      decision_items: [],
    },
  };
}

const meta = {
  title: "Organisms/BriefingCard",
  component: BriefingCard,
  tags: ["autodocs"],
  decorators: [
    (Story) => (
      <div className="w-xl">
        <Story />
      </div>
    ),
  ],
  parameters: {
    layout: "centered",
    // Known axe findings, mostly colour contrast (DESIGN.md, "Story tests").
    // Back to "error" once they are fixed.
    a11y: { test: "todo" },
    docs: {
      description: {
        component:
          "The home page recap: one row per finished run since the last briefing, each with its expert, agent name and summary. Three rows show collapsed; “Show all results” opens the list up to about six rows, with scroll arrows once it runs past that. A run that did not complete is badged Failed. The date reads “This morning” for today's briefing, otherwise “Month day”. Renders nothing when there are no runs.",
      },
    },
  },
  args: { briefing: briefing(RUNS.slice(0, 3)) },
} satisfies Meta<typeof BriefingCard>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const SingleRun: Story = {
  args: { briefing: briefing(RUNS.slice(0, 1)) },
};

export const Collapsed: Story = {
  args: { briefing: briefing(RUNS) },
  parameters: {
    docs: {
      description: {
        story:
          "More than three runs: the rest are clipped, out of the tab order, behind “Show all results”.",
      },
    },
  },
};

export const Expanded: Story = {
  args: { briefing: briefing(RUNS) },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    await userEvent.click(
      canvas.getByRole("button", { name: /Show all results \(8\)/ }),
    );
    await expect(
      await canvas.findByRole("button", { name: /Show less/ }),
    ).toBeInTheDocument();
  },
  parameters: {
    docs: {
      description: {
        story:
          "Opened: the list caps at about six rows and scrolls, with a scroll-down arrow over the last row.",
      },
    },
  },
};

export const FailedRun: Story = {
  args: {
    briefing: briefing([
      RUNS[0],
      runItem(1, {
        expert_name: "Jules",
        expert_avatar_url: MANAGED_IDENTITIES[1].url,
        agent_name: "Invoice chaser",
        status: "FAILED",
        summary: "Could not reach the accounting API after 3 retries.",
      }),
      RUNS[2],
    ]),
  },
};

export const WithoutSummary: Story = {
  args: {
    briefing: briefing([
      runItem(0, {
        expert_name: "Maria",
        expert_avatar_url: MANAGED_IDENTITIES[0].url,
        agent_name: "Lead finder",
        summary: null,
      }),
      runItem(1, {
        expert_name: null,
        agent_name: "Invoice chaser",
        summary: null,
      }),
    ]),
  },
  parameters: {
    docs: {
      description: {
        story:
          "With no summary the expert's name is the whole subtitle; with neither, the row is just the agent name.",
      },
    },
  },
};

export const WithoutLinks: Story = {
  args: {
    briefing: briefing([
      runItem(0, { agent_name: "Lead finder", link: null }),
      runItem(1, {
        agent_name: "Invoice chaser",
        link: "https://example.com/not-internal",
      }),
    ]),
  },
  parameters: {
    docs: {
      description: {
        story:
          "Rows without a safe internal link (missing, absolute or `javascript:`) render as plain rows with no arrow.",
      },
    },
  },
};

export const LongContent: Story = {
  args: {
    briefing: briefing([
      runItem(0, {
        expert_name: "Maria",
        expert_avatar_url: MANAGED_IDENTITIES[0].url,
        agent_name:
          "Quarterly revenue reconciliation across every regional subsidiary ledger",
        summary:
          "Matched 1,204 of 1,210 transactions across the EU, UK and US ledgers. The six left over are all foreign-exchange rounding differences under one euro, listed in the attached sheet for a quick sign-off before the close.",
      }),
    ]),
  },
};

export const Empty: Story = {
  args: { briefing: briefing([]) },
  parameters: {
    docs: {
      description: {
        story:
          "A briefing with no runs renders nothing; callers gate on `hasRecapContent` to show their own fallback.",
      },
    },
  },
};
