import type { Meta, StoryObj } from "@storybook/nextjs";
import { delay, http, HttpResponse } from "msw";
import { fn } from "storybook/test";
import { getGetV1GetExecutionDetailsMockHandler200 } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import type { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import { WorkOutputSheet } from "./WorkOutputSheet";

const GRAPH_ID = "graph-weekly-report";
const EXECUTION_ID = "run-2026-10-01";
const RUN_LINK =
  "/library/agents/library-agent-1?activeTab=runs&activeItem=run-2026-10-01";
const IMAGE_URL = "https://cdn.example.com/storybook/weekly-signups.svg";
const BROKEN_IMAGE_URL = "https://cdn.example.com/storybook/expired.png";

const CHART_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360" viewBox="0 0 640 360"><rect width="640" height="360" fill="#fafafa"/><g fill="#a855f7"><rect x="60" y="220" width="60" height="100"/><rect x="160" y="180" width="60" height="140"/><rect x="260" y="140" width="60" height="180"/><rect x="360" y="100" width="60" height="220"/><rect x="460" y="60" width="60" height="260"/></g><line x1="40" y1="320" x2="600" y2="320" stroke="#71717a" stroke-width="2"/></svg>`;

function execution(outputs: Record<string, unknown[]>): GraphExecution {
  return {
    id: EXECUTION_ID,
    user_id: "user-1",
    graph_id: GRAPH_ID,
    graph_version: 1,
    inputs: {},
    credential_inputs: null,
    nodes_input_masks: null,
    preset_id: null,
    status: "COMPLETED",
    started_at: new Date("2026-10-01T09:00:00Z"),
    ended_at: new Date("2026-10-01T09:02:30Z"),
    stats: null,
    outputs,
  };
}

function outputsHandler(outputs: Record<string, unknown[]>) {
  return getGetV1GetExecutionDetailsMockHandler200(execution(outputs));
}

const SIGNUPS_TABLE = [
  { week: "2026-09-01", signups: 412, activated: 287, channel: "Organic" },
  { week: "2026-09-08", signups: 468, activated: 301, channel: "Organic" },
  { week: "2026-09-15", signups: 523, activated: 344, channel: "Paid" },
  { week: "2026-09-22", signups: 497, activated: 352, channel: "Referral" },
];

const WIDE_TABLE = Array.from({ length: 150 }, (_, row) =>
  Object.fromEntries(
    Array.from({ length: 24 }, (_, column) => [
      `metric_${column + 1}`,
      row * 10 + column,
    ]),
  ),
);

const REPORT_MARKDOWN = `# Weekly report

Signups grew **6%** week over week, led by referrals.

## Highlights

- 497 new signups
- 352 activated accounts
- Referral share up to 21%

| Channel  | Signups |
| -------- | ------- |
| Organic  | 210     |
| Paid     | 182     |
| Referral | 105     |
`;

const meta = {
  title: "Organisms/WorkOutputSheet",
  component: WorkOutputSheet,
  tags: ["autodocs"],
  parameters: {
    layout: "fullscreen",
    a11y: { test: "error" },
    msw: { handlers: [outputsHandler({ result: [SIGNUPS_TABLE] })] },
    docs: {
      description: {
        component:
          "Right-hand sheet that previews one run's output by its classified type: a table with CSV export, rendered markdown for documents, or an image. Unknown types, failed loads and unsafe or broken images fall back to a link to the full run.",
      },
      story: { inline: false, iframeHeight: 640 },
    },
  },
  args: {
    open: true,
    onOpenChange: fn(),
    title: "Weekly Report",
    outputType: "table",
    graphId: GRAPH_ID,
    executionId: EXECUTION_ID,
    runLink: RUN_LINK,
  },
} satisfies Meta<typeof WorkOutputSheet>;

export default meta;
type Story = StoryObj<typeof meta>;

export const TableOutput: Story = {};

export const TruncatedTable: Story = {
  parameters: {
    msw: { handlers: [outputsHandler({ result: [WIDE_TABLE] })] },
  },
};

export const DocumentOutput: Story = {
  args: { outputType: "doc" },
  parameters: {
    msw: { handlers: [outputsHandler({ report: [REPORT_MARKDOWN] })] },
  },
};

export const ImageOutput: Story = {
  args: { outputType: "image", title: "Weekly signups chart" },
  parameters: {
    msw: {
      handlers: [
        outputsHandler({ chart: [IMAGE_URL] }),
        http.get(
          IMAGE_URL,
          () =>
            new HttpResponse(CHART_SVG, {
              headers: { "Content-Type": "image/svg+xml" },
            }),
        ),
      ],
    },
  },
};

export const BrokenImage: Story = {
  args: { outputType: "image", title: "Weekly signups chart" },
  parameters: {
    msw: {
      handlers: [
        outputsHandler({ chart: [BROKEN_IMAGE_URL] }),
        http.get(
          BROKEN_IMAGE_URL,
          () => new HttpResponse(null, { status: 404 }),
        ),
      ],
    },
  },
};

export const UnknownOutput: Story = {
  args: { outputType: "unknown" },
};

export const NoPreviewWithoutRunLink: Story = {
  args: { outputType: "unknown", runLink: null },
};

export const Loading: Story = {
  parameters: {
    msw: {
      handlers: [
        http.get("*/api/graphs/:graphId/executions/:graphExecId", async () => {
          await delay("infinite");
          return HttpResponse.json({});
        }),
      ],
    },
  },
};

export const LoadError: Story = {
  parameters: {
    msw: {
      handlers: [
        http.get("*/api/graphs/:graphId/executions/:graphExecId", () =>
          HttpResponse.json(
            { detail: "Failed to load run outputs" },
            { status: 500 },
          ),
        ),
      ],
    },
  },
};
