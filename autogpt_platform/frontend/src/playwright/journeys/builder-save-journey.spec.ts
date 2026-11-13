import type { Page } from "@playwright/test";
import { BuildPage, createUniqueAgentName } from "../pages/build.page";
import { expect, test } from "./fixtures";

test.describe("builder save journey", () => {
  test.use({
    expectedResponses: [
      { status: 400, url: "/api/proxy/api/graphs", method: "POST" },
    ],
  });

  test("saving with a required field left empty names the field, saves nothing, and saves once it is filled", async ({
    page,
    journeyUser: _user,
  }) => {
    const buildPage = new BuildPage(page);
    const agentName = createUniqueAgentName("Journey unnamed output");
    await buildPage.open();
    await buildPage.addBlockByClick("Agent Output");
    await buildPage.waitForNodeOnCanvas(1);
    const outputNode = buildPage.getNodeLocatorByTitle("Agent Output");

    await buildPage.saveAgent(agentName, "Builder save journey");

    await expect(page.getByText("Couldn't save your workflow")).toBeVisible();
    await expect(
      page.getByText(/Agent Output › Name: This field is required/),
    ).toBeVisible();
    await expect(page.getByText(/has no attribute/)).toHaveCount(0);
    await expect(
      outputNode.getByText("This field is required"),
      "the empty field should be marked on its node",
    ).toBeVisible();
    await expect(page).not.toHaveURL(/flowID=/);
    expect(await libraryAgentNames(page, agentName)).toEqual([]);

    await page.keyboard.press("Escape");
    await outputNode
      .getByRole("textbox", { name: "Name", exact: true })
      .fill("greeting");
    await buildPage.saveAgent(agentName, "Builder save journey");
    await buildPage.waitForSaveComplete();

    await expect(outputNode.getByText("This field is required")).toHaveCount(0);
    expect(await libraryAgentNames(page, agentName)).toEqual([agentName]);
  });
});

async function libraryAgentNames(page: Page, name: string) {
  const response = await page.request.get("/api/proxy/api/library/agents", {
    params: { search_term: name },
  });
  expect(response.ok(), `GET library agents: ${response.status()}`).toBe(true);
  const body = (await response.json()) as { agents: { name: string }[] };
  return body.agents.map((agent) => agent.name);
}
