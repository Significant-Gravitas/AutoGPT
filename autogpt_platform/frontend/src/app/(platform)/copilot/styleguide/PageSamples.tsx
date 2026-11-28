import { Page } from "@/app/(platform)/pages/components/Page/Page";
import type { ResolvedPage } from "@/app/api/__generated__/models/resolvedPage";

const IDEAS = { library_agent_id: "ideas", workflow_name: "Keyword ideas" };
const DRAFT = { library_agent_id: "draft", workflow_name: "Draft a post" };
const NADIA = { expert_id: "nadia", expert_name: "Nadia" };
const RUN = { status: "COMPLETED", cost_cents: 4 } as const;

const SAMPLE = {
  id: "sample",
  title: "LinkedIn posts",
  revision: 3,
  chat_session_id: null,
  has_previous: true,
  sections: [
    {
      kind: "text",
      index: 0,
      heading: "This week",
      body: "Pick a keyword and draft a post from it.",
      tone: "takeaway",
    },
    {
      kind: "latest_result",
      index: 1,
      ...IDEAS,
      run_id: "r1",
      run_status: "COMPLETED",
      finished_at: "2026-11-21T09:00:00Z",
      blocks: [
        {
          type: "table",
          title: "Keywords",
          columns: ["Keyword", "Searches"],
          rows: [
            ["remote work", "12,400"],
            ["four-day week", "8,100"],
          ],
        },
      ],
    },
    {
      kind: "run_form",
      index: 2,
      ...DRAFT,
      button_label: "Draft it",
      fields: [
        {
          input: "keyword",
          label: "Keyword",
          default: "remote work",
          input_schema: { type: "string" },
          required: true,
        },
      ],
    },
    {
      kind: "quick_runs",
      index: 3,
      ...IDEAS,
      templates: [
        { id: "t1", name: "Tech topics" },
        { id: "t2", name: "HR topics" },
      ],
    },
    {
      kind: "run_history",
      index: 4,
      ...DRAFT,
      columns: ["Title"],
      rows: [
        {
          run_id: "d1",
          status: "COMPLETED",
          started_at: "2026-11-21T09:05:00Z",
          cells: ["Why remote teams write more"],
        },
      ],
    },
    {
      kind: "run_numbers",
      index: 5,
      ...DRAFT,
      period: "7d",
      block: {
        type: "stats",
        items: [
          { label: "Runs", value: "9" },
          { label: "Succeeded", value: "89%" },
          { label: "Spent", value: "$0.31" },
        ],
      },
    },
    {
      kind: "recent_runs",
      index: 6,
      ...DRAFT,
      runs: [
        {
          id: "d1",
          ...RUN,
          started_at: "2026-11-21T09:05:00Z",
          summary: "Drafted a 180-word post.",
        },
      ],
    },
    {
      kind: "schedule",
      index: 7,
      ...IDEAS,
      is_scheduled: true,
      next_run: "2026-11-24T08:00:00Z",
    },
    {
      kind: "expert_status",
      index: 8,
      ...NADIA,
      status: "working",
      detail: "Running Competitor watch",
      spend_cents: 120,
    },
    {
      kind: "expert_learned",
      index: 9,
      ...NADIA,
      facts: [{ fact: "Acme cut its prices by 10% in November." }],
    },
    { kind: "waiting", index: 10, scope_name: null, reviews: [] },
    {
      kind: "notice",
      index: 11,
      section_kind: "run_form",
      reason: "stale",
      message: "This section no longer fits its workflow. Ask Otto to fix it.",
    },
  ],
} as unknown as ResolvedPage;

const NO_OP_EDITING = {
  onMove() {},
  onRemove() {},
  onChangeWithOtto() {},
};

export function PageSamples() {
  return (
    <div className="flex max-w-3xl flex-col gap-8">
      <Page page={SAMPLE} />
      <Page
        page={{ ...SAMPLE, sections: SAMPLE.sections.slice(0, 2) }}
        editing={NO_OP_EDITING}
      />
    </div>
  );
}
