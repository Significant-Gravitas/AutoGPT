import fs from "fs";
import type { Page, TestInfo } from "@playwright/test";
// importAgentFromFile loads this lazily, which only works once it's loaded.
import "../pages/build.page";
import {
  assertRunOutputValue,
  clickRunButton,
  getActiveItemId,
  importAgentFromFile,
  waitForAgentPageLoad,
} from "../pages/library.page";
import { creditsSince, snapshotCredits } from "./credits";
import { expect, test } from "./fixtures";

const CALCULATOR_BLOCK_ID = "b1ab9b19-67a6-406d-abf5-2dba76d00c79";
const MATCH_TEXT_PATTERN_BLOCK_ID = "3060088f-6ed9-4928-9ba7-9c92823a7ccd";
const AGENT_OUTPUT_BLOCK_ID = "363ae599-353e-4804-937e-b2ee3cef3da4";

// Credits charged per node execution. CI's journeys leg bills every node so a
// run's cost is visible; a hosted stack bills one credit per 100 executions,
// which a fresh user never reaches.
const COST_PER_NODE = Number(process.env.PLAYWRIGHT_JOURNEY_COST_PER_NODE ?? 0);

const TERMINAL_STATUS = /^(COMPLETED|FAILED|TERMINATED)$/;

test.describe("run correctness journey", () => {
  test("a run completes, shows its output in the run view and the Library, and costs exactly its block cost", async ({
    page,
    journeyUser: _user,
  }, testInfo) => {
    const agentName = uniqueName("Journey multiply");
    const outputName = "answer";
    const agent = await importAgent(page, testInfo, {
      name: agentName,
      nodes: [
        node("multiply", CALCULATOR_BLOCK_ID, {
          operation: "Multiply",
          a: 6,
          b: 7,
          round_result: false,
        }),
        node("output", AGENT_OUTPUT_BLOCK_ID, { name: outputName }),
      ],
      links: [link("multiply", "result", "output", "value")],
    });
    const before = await snapshotCredits(page);

    const runID = await startRunFromLibrary(page, agent.url);

    await expect
      .poll(() => getRunStatus(page, agent.graphID, runID), {
        message: "the run should reach COMPLETED",
        timeout: 30_000,
      })
      .toBe("COMPLETED");
    await assertRunOutputValue(page, outputName, "42");

    await page.goto(agent.url);
    await waitForAgentPageLoad(page, agentName);
    await expect(
      runListItem(page, runID),
      "the Library run list should show the run",
    ).toBeVisible();

    await expect
      .poll(
        async () => (await creditsSince(page, before, runID)).chargedToRun,
        {
          message: "the run should be charged its block cost",
        },
      )
      .toBe(COST_PER_NODE * 2);
    const ledger = await creditsSince(page, before, runID);
    expect(
      ledger.balance - before.balance,
      `the balance should move by exactly the new ledger entries: ${JSON.stringify(ledger.added)}`,
    ).toBe(ledger.netChange);
  });

  test("a run whose node fails ends instead of hanging", async ({
    page,
    journeyUser: _user,
  }, testInfo) => {
    const { agent, runID } = await runAgentWithFailingNode(page, testInfo);

    await expect
      .poll(() => getRunStatus(page, agent.graphID, runID), {
        message: "the run should end, not stay QUEUED or RUNNING",
        timeout: 30_000,
      })
      .toMatch(TERMINAL_STATUS);
  });

  test("a run whose node fails shows the user which step failed and why", async ({
    page,
    journeyUser: _user,
  }, testInfo) => {
    // A node error with nothing wired to its error output doesn't fail the
    // run (backend/executor/manager.py), so the run reports COMPLETED. Whether
    // it should fail is the open product question of #14275 and SECRT-2840;
    // either way the user must see which step failed and why.
    const { agent, runID } = await runAgentWithFailingNode(page, testInfo);

    await expect
      .poll(() => getRunStatus(page, agent.graphID, runID), {
        message: "the run should end as COMPLETED, its node error non-fatal",
        timeout: 30_000,
      })
      .toBe("COMPLETED");

    const failures = page.getByTestId("run-failures");
    await expect(
      failures,
      "the run view should say that a step failed",
    ).toContainText("1 step failed");
    await expect(
      failures,
      "the run view should name the step that failed",
    ).toContainText("Match Text Pattern");
    await expect(
      failures,
      "the run view should show the step's error",
    ).toContainText(/Invalid regex pattern/);

    await page.goto(agent.url);
    await waitForAgentPageLoad(page);
    await expect(
      runListItem(page, runID),
      "the Library run list should mark the run as having a failed step",
    ).toContainText("1 step failed");
  });
});

async function runAgentWithFailingNode(page: Page, testInfo: TestInfo) {
  const agent = await importAgent(page, testInfo, {
    name: uniqueName("Journey bad pattern"),
    nodes: [
      node("match", MATCH_TEXT_PATTERN_BLOCK_ID, {
        text: "journey",
        match: "(",
        data: "matched",
      }),
      node("output", AGENT_OUTPUT_BLOCK_ID, { name: "matched" }),
    ],
    links: [link("match", "positive", "output", "value")],
  });
  const runID = await startRunFromLibrary(page, agent.url);
  return { agent, runID };
}

interface GraphSpec {
  name: string;
  nodes: ReturnType<typeof node>[];
  links: ReturnType<typeof link>[];
}

function node(id: string, blockID: string, input: Record<string, unknown>) {
  return {
    id,
    block_id: blockID,
    input_default: input,
    metadata: { position: { x: 0, y: 0 } },
    input_links: [],
    output_links: [],
  };
}

function link(source: string, output: string, sink: string, input: string) {
  return {
    source_id: source,
    source_name: output,
    sink_id: sink,
    sink_name: input,
  };
}

function uniqueName(prefix: string) {
  return `${prefix} ${Date.now()}`;
}

async function importAgent(page: Page, testInfo: TestInfo, graph: GraphSpec) {
  const nodes = graph.nodes.map((graphNode, index) => ({
    ...graphNode,
    metadata: { position: { x: 120 + index * 400, y: 160 } },
  }));
  const filePath = testInfo.outputPath(`${graph.nodes[0].id}-agent.json`);
  fs.writeFileSync(
    filePath,
    JSON.stringify({
      name: graph.name,
      description: "Built by the run correctness journey",
      is_active: true,
      nodes,
      links: graph.links,
    }),
  );
  await importAgentFromFile(page, filePath, graph.name);
  return findLibraryAgent(page, graph.name);
}

async function findLibraryAgent(page: Page, name: string) {
  const response = await page.request.get("/api/proxy/api/library/agents", {
    params: { search_term: name },
  });
  expect(response.ok(), `GET library agents: ${response.status()}`).toBe(true);
  const body = (await response.json()) as {
    agents: { id: string; graph_id: string; name: string }[];
  };
  const agent = body.agents.find((candidate) => candidate.name === name);
  if (!agent) throw new Error(`library agent "${name}" should exist`);
  return { graphID: agent.graph_id, url: `/library/agents/${agent.id}` };
}

async function startRunFromLibrary(page: Page, agentURL: string) {
  await page.goto(agentURL);
  await waitForAgentPageLoad(page);
  await clickRunButton(page);
  await expect
    .poll(() => getActiveItemId(page), {
      message: "starting a run should open it in the run view",
    })
    .toBeTruthy();
  return String(getActiveItemId(page));
}

async function getRunStatus(page: Page, graphID: string, runID: string) {
  const response = await page.request.get(
    `/api/proxy/api/graphs/${graphID}/executions/${runID}`,
  );
  expect(response.ok(), `GET execution: ${response.status()}`).toBe(true);
  const body = (await response.json()) as { status: string };
  return body.status;
}

function runListItem(page: Page, runID: string) {
  return page.getByTestId(`library-run-item-${runID}`);
}
