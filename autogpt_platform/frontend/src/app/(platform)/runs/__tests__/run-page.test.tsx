import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it, vi } from "vitest";

import { getGetV2GetLibraryAgentByGraphIdResponseMock } from "@/app/api/__generated__/endpoints/library/library.msw";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";

import RunPage from "../[graphId]/[runId]/page";

const GRAPH = "graph-1";
const BASE = `/api/proxy/api/graphs/${GRAPH}`;

vi.mock("next/navigation", async (importOriginal) => ({
  ...(await importOriginal<typeof import("next/navigation")>()),
  useParams: () => ({ graphId: GRAPH, runId: "run-1" }),
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => `/runs/${GRAPH}/run-1`,
  useSearchParams: () => new URLSearchParams(),
}));

function block(
  id: string,
  name: string,
  category: string,
  uiType = "Standard",
) {
  return {
    id,
    name,
    categories: [{ category, description: "" }],
    uiType,
    inputSchema: {},
    outputSchema: {},
    costs: [],
    description: "",
    contributors: [],
    staticOutput: false,
  };
}

function execution(
  id: string,
  blockID: string,
  status: string,
  start: string,
  end: string,
  output: Record<string, unknown[]>,
) {
  return {
    user_id: "u",
    graph_id: GRAPH,
    graph_version: 1,
    graph_exec_id: "run-1",
    node_exec_id: id,
    node_id: `node-${id}`,
    block_id: blockID,
    status,
    input_data: { topic: "Weekly sales" },
    output_data: output,
    add_time: start,
    queue_time: null,
    start_time: start,
    end_time: end,
  };
}

function serve() {
  const run = {
    id: "run-1",
    user_id: "u",
    graph_id: GRAPH,
    graph_version: 1,
    inputs: { topic: "Weekly sales" },
    outputs: {},
    credential_inputs: {},
    nodes_input_masks: {},
    preset_id: null,
    status: "FAILED",
    started_at: "2026-11-15T09:00:00Z",
    ended_at: "2026-11-15T09:00:05Z",
    trigger_source: "schedule",
    stats: { cost: 12 },
    node_executions: [
      execution(
        "s1",
        "b-input",
        "COMPLETED",
        "2026-11-15T09:00:00Z",
        "2026-11-15T09:00:01Z",
        { result: ["Weekly sales"] },
      ),
      execution(
        "s2",
        "b-ai",
        "FAILED",
        "2026-11-15T09:00:01Z",
        "2026-11-15T09:00:05Z",
        { error: ["The model account has no credits left"] },
      ),
    ],
  };
  server.use(
    http.get("/api/proxy/api/library/agents/by-graph/:id", () =>
      HttpResponse.json({
        ...getGetV2GetLibraryAgentByGraphIdResponseMock(),
        graph_id: GRAPH,
        graph_version: 1,
        name: "Sales digest",
        input_schema: { properties: {}, required: [] },
        credentials_input_schema: { properties: {} },
        trigger_setup_info: null,
      }),
    ),
    http.get(`${BASE}/executions/:execID`, () => HttpResponse.json(run)),
    http.get(BASE, () =>
      HttpResponse.json({ name: "Sales digest", nodes: [] }),
    ),
    http.get("/api/proxy/api/builder/blocks/batch", () =>
      HttpResponse.json([
        block("b-input", "AgentInputBlock", "INPUT", "Input"),
        block("b-ai", "AITextGeneratorBlock", "AI"),
      ]),
    ),
  );
}

describe("Run page", () => {
  it("shows the run with its figures and its blocks on one axis", async () => {
    serve();
    render(<RunPage />);

    expect(
      await screen.findByText("Sales digest", { selector: "h1" }),
    ).toBeDefined();
    expect(screen.getByText(/^Scheduled run · /)).toBeDefined();
    expect(
      document.querySelector(".ui-ard-head .ui-ard-status")?.textContent,
    ).toBe("Failed");
    const figures = Array.from(document.querySelectorAll(".ui-ard-stat")).map(
      (card) => card.textContent,
    );
    expect(figures).toEqual(["Time5.0s", "Cost$0.120", "Blocks2", "Failed1"]);
    await screen.findByText("AI Text Generator", {
      selector: ".ui-ard-row__name",
    });
    const rows = Array.from(document.querySelectorAll("[data-step]"));
    expect(rows.map((row) => row.getAttribute("aria-label"))).toEqual([
      "Agent Input – Input and output, 1.0s",
      "AI Text Generator – AI, failed",
    ]);
  });

  it("opens a failed block beside the run with its error and readable values", async () => {
    serve();
    render(<RunPage />);

    await screen.findByText("AI Text Generator", {
      selector: ".ui-ard-row__name",
    });
    await userEvent.click(
      document.querySelector<HTMLElement>('[data-step="s2"]')!,
    );

    expect(
      await screen.findByText("The model account has no credits left"),
    ).toBeDefined();
    const panel = screen.getByLabelText("Open block runs");
    expect(panel.textContent).toContain("Topic");
    expect(panel.textContent).toContain("Weekly sales");
    expect(panel.querySelector("pre")).toBeNull();
  });
});
