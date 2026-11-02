import { getGetV1ListAllExecutionsMockHandler200 } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import {
  getGetV2GetLibraryAgentResponseMock200,
  getGetV2ListLibraryAgentsResponseMock200,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { useState } from "react";
import { expect, userEvent, within } from "storybook/test";
import { AccountMenuActivityRow } from "../AccountMenu/components/AccountMenuActivityRow";

const graphID = "abcdef01-late-workflow";
const workflow = getGetV2GetLibraryAgentResponseMock200({
  id: "library-late-workflow",
  graph_id: graphID,
  name: "Daily research",
  description: "Daily research summary",
});
const firstWorkflow = {
  ...workflow,
  id: "library-first-workflow",
  graph_id: "first-workflow",
  name: "Weekly summary",
};
const firstPageWorkflows = Array.from({ length: 100 }, (_, index) => ({
  ...firstWorkflow,
  id: `library-${index}`,
  graph_id: `workflow-${index}`,
}));
const execution: GraphExecutionMeta = {
  id: "run-1",
  user_id: "user-1",
  graph_id: graphID,
  graph_version: 1,
  inputs: {},
  credential_inputs: {},
  nodes_input_masks: {},
  preset_id: null,
  status: "RUNNING",
  started_at: new Date(),
  ended_at: null,
  stats: null,
};
let attempts = 0;

function ActivityStory() {
  const [queryClient] = useState(
    () => new QueryClient({ defaultOptions: { queries: { retryDelay: 0 } } }),
  );
  return (
    <QueryClientProvider client={queryClient}>
      <BackendAPIProvider>
        <div className="w-64">
          <AccountMenuActivityRow />
        </div>
      </BackendAPIProvider>
    </QueryClientProvider>
  );
}

const meta = {
  title: "Layout/Workflow Activity",
  component: AccountMenuActivityRow,
  render: ActivityStory,
  beforeEach() {
    attempts = 0;
  },
  parameters: {
    msw: {
      handlers: [
        getGetV1ListAllExecutionsMockHandler200([execution]),
        http.get("/api/proxy/api/library/agents", ({ request }) => {
          const currentPage = Number(
            new URL(request.url).searchParams.get("page"),
          );
          if (currentPage === 2 && ++attempts <= 3)
            return HttpResponse.json(
              { detail: "Temporarily unavailable" },
              { status: 503 },
            );
          return HttpResponse.json(
            getGetV2ListLibraryAgentsResponseMock200({
              agents: currentPage === 1 ? firstPageWorkflows : [workflow],
              pagination: {
                current_page: currentPage,
                total_items: 101,
                total_pages: 2,
                page_size: 100,
              },
            }),
          );
        }),
      ],
    },
  },
} satisfies Meta<typeof AccountMenuActivityRow>;

export default meta;
type Story = StoryObj<typeof meta>;

export const UnavailableDetails: Story = {};

export const RetryDetails: Story = {
  play: async ({ canvasElement }) => {
    const ui = within(canvasElement.ownerDocument.body);
    await userEvent.click(ui.getByTestId("account-menu-activity-trigger"));
    const retry = await ui.findByRole("button", { name: "Try Again" });
    await expect(ui.getByText("Workflow abcdef01")).toBeVisible();
    await userEvent.click(retry);
    await expect(
      await ui.findByRole("link", { name: /^Daily research/ }),
    ).toHaveAttribute(
      "href",
      "/library/agents/library-late-workflow?activeTab=runs&activeItem=run-1",
    );
  },
};
