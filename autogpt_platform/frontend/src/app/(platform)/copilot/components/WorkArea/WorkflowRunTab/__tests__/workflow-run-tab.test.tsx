import { http, HttpResponse } from "msw";
import { QueryClient } from "@tanstack/react-query";
import { afterEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { getGetV2GetLibraryAgentByGraphIdResponseMock } from "@/app/api/__generated__/endpoints/library/library.msw";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { useWorkAreaStore } from "../../../../workArea";
import { WorkflowRunTab } from "../WorkflowRunTab";
import type { WorkflowTab } from "../useWorkflowRun";

const GRAPH = "graph-1";
const BASE = `/api/proxy/api/graphs/${GRAPH}`;

const BLOCKS = [
  block("b-fetch", "FetchPageBlock", "SEARCH"),
  block("b-sum", "SummariseBlock", "AI"),
  block("b-post", "PostToSlackBlock", "COMMUNICATION"),
];

function block(id: string, name: string, category: string) {
  return {
    id,
    name,
    categories: [{ category, description: "" }],
    uiType: "Standard",
    inputSchema: {},
    outputSchema: {},
    costs: [],
    description: "",
    contributors: [],
    staticOutput: false,
  };
}

interface Step {
  id: string;
  blockID: string;
  status: string;
  start: string | null;
  end: string | null;
  output?: Record<string, unknown[]>;
}

function nodeExecution(step: Step) {
  return {
    user_id: "u",
    graph_id: GRAPH,
    graph_version: 1,
    graph_exec_id: "run-1",
    node_exec_id: step.id,
    node_id: `node-${step.id}`,
    block_id: step.blockID,
    status: step.status,
    input_data: { url: "https://example.com" },
    output_data: step.output ?? { result: ["ok"] },
    add_time: step.start ?? "2026-11-15T09:00:00Z",
    queue_time: null,
    start_time: step.start,
    end_time: step.end,
  };
}

function runMeta(id: string, status: string, extra: object = {}) {
  return {
    id,
    user_id: "u",
    graph_id: GRAPH,
    graph_version: 1,
    inputs: {},
    credential_inputs: {},
    nodes_input_masks: {},
    preset_id: null,
    status,
    started_at: "2026-11-15T09:00:00Z",
    ended_at: status === "RUNNING" ? null : "2026-11-15T09:00:08Z",
    trigger_source: "manual",
    stats: null,
    ...extra,
  };
}

function serve(runs: object[], detail: object | null) {
  server.use(
    http.get("/api/proxy/api/library/agents/by-graph/:id", () =>
      HttpResponse.json({
        ...getGetV2GetLibraryAgentByGraphIdResponseMock(),
        graph_id: GRAPH,
        graph_version: 1,
        name: "Morning digest",
        input_schema: { properties: {}, required: [] },
        credentials_input_schema: { properties: {} },
        trigger_setup_info: null,
      }),
    ),
    http.get(`${BASE}/executions`, () =>
      HttpResponse.json({
        executions: runs,
        pagination: {
          total_items: runs.length,
          total_pages: 1,
          current_page: 1,
          page_size: 20,
        },
      }),
    ),
    http.get(`${BASE}/executions/:execID`, () =>
      detail
        ? HttpResponse.json(detail)
        : new HttpResponse(null, { status: 404 }),
    ),
    http.get(BASE, () =>
      HttpResponse.json({ name: "Morning digest", nodes: [] }),
    ),
    http.get("/api/proxy/api/builder/blocks/batch", () =>
      HttpResponse.json(BLOCKS),
    ),
  );
}

function renderTab(executionID: string | null = "run-1") {
  const tab: WorkflowTab = {
    key: `workflow:${GRAPH}`,
    kind: "workflow",
    graphID: GRAPH,
    graphVersion: 1,
    executionID,
    title: "Morning digest",
  };
  return render(<WorkflowRunTab tab={tab} />);
}

/** The block runs as the Run Detail block draws them: one row button each. */
async function rows() {
  await screen.findByText("Post To Slack", { selector: ".ui-ard-row__name" });
  return Array.from(document.querySelectorAll<HTMLElement>("[data-step]"));
}

function runStatus() {
  return document.querySelector(".ui-ard-head .ui-ard-status")?.textContent;
}

describe("Workflow run tab", () => {
  it("lists a finished run's blocks in the order they started", async () => {
    // The API sends them out of order; the trace sorts by start time.
    const steps = [
      {
        id: "s3",
        blockID: "b-post",
        status: "COMPLETED",
        start: "2026-11-15T09:00:06Z",
        end: "2026-11-15T09:00:08Z",
      },
      {
        id: "s1",
        blockID: "b-fetch",
        status: "COMPLETED",
        start: "2026-11-15T09:00:00Z",
        end: "2026-11-15T09:00:02Z",
      },
      {
        id: "s2",
        blockID: "b-sum",
        status: "COMPLETED",
        start: "2026-11-15T09:00:02Z",
        end: "2026-11-15T09:00:06Z",
      },
    ];
    serve([runMeta("run-1", "COMPLETED")], {
      ...runMeta("run-1", "COMPLETED"),
      node_executions: steps.map(nodeExecution),
    });
    renderTab();

    const texts = (await rows()).map((row) => row.textContent ?? "");
    expect(texts).toHaveLength(3);
    expect(texts[0]).toContain("Fetch Page");
    expect(texts[1]).toContain("Summarise");
    expect(texts[2]).toContain("Post To Slack");
    expect(texts[1]).toContain("4.0s");
    expect(runStatus()).toBe("Done");
    expect(
      screen.getByText("Morning digest", { selector: "h1" }),
    ).toBeDefined();
    expect(
      screen.getByText("Open full page").closest("a")?.getAttribute("href"),
    ).toBe("/runs/graph-1/run-1");
  });

  it("shows a failed block's error when its row opens", async () => {
    const steps = [
      {
        id: "s1",
        blockID: "b-fetch",
        status: "COMPLETED",
        start: "2026-11-15T09:00:00Z",
        end: "2026-11-15T09:00:02Z",
      },
      {
        id: "s2",
        blockID: "b-post",
        status: "FAILED",
        start: "2026-11-15T09:00:02Z",
        end: "2026-11-15T09:00:03Z",
        output: { error: ["Slack said the channel does not exist"] },
      },
    ];
    serve([runMeta("run-1", "FAILED")], {
      ...runMeta("run-1", "FAILED"),
      node_executions: steps.map(nodeExecution),
    });
    renderTab();

    const failed = (await rows()).find((row) =>
      row.textContent?.includes("Post To Slack"),
    )!;
    expect(failed.getAttribute("aria-label")).toContain("failed");
    expect(
      screen.queryByText("Slack said the channel does not exist"),
    ).toBeNull();
    await userEvent.click(failed);

    expect(
      await screen.findByText("Slack said the channel does not exist"),
    ).toBeDefined();
    // The block run's input, drawn to be read rather than as JSON.
    expect(
      screen
        .getByText(/example\.com/)
        .closest("a")
        ?.getAttribute("href"),
    ).toBe("https://example.com");
    await userEvent.click(screen.getByText("Back to the run"));
    expect((await rows()).length).toBe(2);
  });

  it("offers Run now when the workflow has no runs", async () => {
    serve([], null);
    renderTab(null);

    expect(
      await screen.findByText("This workflow has not run yet."),
    ).toBeDefined();
    const buttons = await screen.findAllByRole("button", { name: "Run now" });
    expect(buttons.length).toBeGreaterThan(0);
  });

  it("shows a block that is still running as a running row", async () => {
    const started = new Date(Date.now() - 3000).toISOString();
    const steps = [
      {
        id: "s1",
        blockID: "b-fetch",
        status: "COMPLETED",
        start: started,
        end: new Date(Date.now() - 2000).toISOString(),
      },
      {
        id: "s2",
        blockID: "b-sum",
        status: "RUNNING",
        start: new Date(Date.now() - 2000).toISOString(),
        end: null,
      },
    ];
    const run = runMeta("run-1", "RUNNING", { started_at: started });
    serve([run], { ...run, node_executions: steps.map(nodeExecution) });
    renderTab();

    await screen.findByText("Summarise", { selector: ".ui-ard-row__name" });
    const running = document.querySelector<HTMLElement>('[data-step="s2"]')!;
    expect(running.getAttribute("aria-label")).toMatch(/Summarise .*running$/);
    expect(
      running.querySelector(".ui-ard-bar")?.hasAttribute("data-running"),
    ).toBe(true);
    expect(runStatus()).toBe("Running");
    expect(screen.getByText("Stop", { selector: "button" })).toBeDefined();
  });

  describe("a run that cannot be read", () => {
    const real = useWorkAreaStore.getState().openWorkflowRun;
    afterEach(() => useWorkAreaStore.setState({ openWorkflowRun: real }));

    it.each([404, 422, 500])(
      "goes back to the list of runs quickly after a %i",
      async (status) => {
        const openWorkflowRun = vi.fn();
        useWorkAreaStore.setState({ openWorkflowRun });
        serve([runMeta("run-2", "COMPLETED")], null);
        server.use(
          http.get(`${BASE}/executions/:execID`, () =>
            HttpResponse.json({ detail: "Not found" }, { status }),
          ),
        );
        // The app's own retry defaults, which the test client turns off.
        render(
          <WorkflowRunTab
            tab={{
              key: `workflow:${GRAPH}`,
              kind: "workflow",
              graphID: GRAPH,
              graphVersion: 1,
              executionID: "missing",
              title: "Morning digest",
            }}
          />,
          { queryClient: new QueryClient() },
        );

        await vi.waitFor(
          () =>
            expect(openWorkflowRun).toHaveBeenCalledWith(
              expect.objectContaining({ executionID: null }),
            ),
          { timeout: 2500 },
        );
      },
    );
  });
});
