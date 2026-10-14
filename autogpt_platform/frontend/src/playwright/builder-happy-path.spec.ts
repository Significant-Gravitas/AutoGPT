import { expect, test } from "./coverage-fixture";
import { E2E_AUTH_STATES } from "./credentials/accounts";
import { BuildPage, SIMPLE_AGENT } from "./pages/build.page";

test.use({ storageState: E2E_AUTH_STATES.builder });

interface SavedGraph {
  nodes: { id: string; block_id: string }[];
  links: {
    source_id: string;
    sink_id: string;
    source_name: string;
    sink_name: string;
  }[];
}

test("builder happy path: user can walk through the builder tutorial and cancel midway, persisting canceled state", async ({
  page,
}) => {
  test.setTimeout(180000);

  const buildPage = new BuildPage(page);
  await buildPage.startTutorial();
  await buildPage.walkWelcomeToBlockMenu();
  await buildPage.walkSearchAndAddCalculator();
  await buildPage.cancelTutorial();

  expect(await buildPage.getTutorialStateFromStorage()).toBe("canceled");
  expect(await buildPage.getNodeCount()).toBeGreaterThanOrEqual(1);
});

test("builder happy path: user can skip the builder tutorial from the welcome step", async ({
  page,
}) => {
  test.setTimeout(60000);

  const buildPage = new BuildPage(page);
  await buildPage.startTutorial();
  await buildPage.skipTutorialFromWelcome();
});

test("builder happy path: user can create a simple agent in builder with core blocks", async ({
  page,
}) => {
  test.setTimeout(120000);

  const buildPage = new BuildPage(page);
  await buildPage.open();
  await buildPage.addSimpleAgentBlocks();

  await expect(buildPage.getNodeLocator()).toHaveCount(2);
  await expect(
    buildPage
      .getNodeLocatorByTitle("Store Value")
      .getByRole("textbox", { name: "Input", exact: true }),
  ).toHaveValue(SIMPLE_AGENT.value);
  await expect(
    buildPage
      .getNodeLocatorByTitle("Agent Output")
      .getByRole("textbox", { name: "Name", exact: true }),
  ).toHaveValue(SIMPLE_AGENT.outputName);
});

test("builder happy path: user can save the created agent", async ({
  page,
}) => {
  test.setTimeout(120000);

  const buildPage = new BuildPage(page);
  const { graphId, graphVersion } =
    await buildPage.createAndSaveSimpleAgent("Smoke Save Agent");

  await expect(page).toHaveURL(/flowID=/);
  expect(await buildPage.isRunButtonEnabled()).toBeTruthy();

  const response = await page.request.get(
    `/api/proxy/api/graphs/${graphId}?version=${graphVersion}`,
  );
  expect(response.ok(), `GET saved graph: ${response.status()}`).toBe(true);
  const graph = (await response.json()) as SavedGraph;
  expect(graph.nodes).toHaveLength(2);
  expect(
    graph.links.map((link) => `${link.source_name} -> ${link.sink_name}`),
    "the saved graph should keep the edge the user drew",
  ).toEqual(["output -> value"]);
});

test("builder happy path: user can run the saved agent from builder and see execution state", async ({
  page,
}) => {
  test.setTimeout(120000);

  const buildPage = new BuildPage(page);
  await buildPage.createAndSaveSimpleAgent("Smoke Run Agent");

  await buildPage.startRun();
  const run = await buildPage.waitForRunToEnd();

  expect(run.status).toBe("COMPLETED");
  expect(run.outputs, "the run should output the stored value").toEqual({
    [SIMPLE_AGENT.outputName]: [SIMPLE_AGENT.value],
  });
  await expect(
    buildPage
      .getNodeLocatorByTitle("Agent Output")
      .getByText("COMPLETED", { exact: true }),
    "the canvas should show the output node's run state",
  ).toBeVisible({ timeout: 15000 });
});
