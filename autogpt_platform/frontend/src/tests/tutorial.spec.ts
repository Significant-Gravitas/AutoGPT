import { expect, TestInfo } from "@playwright/test";
import { test } from "./fixtures";
import { BuildPage } from "./pages/build.page";
import { MonitorPage } from "./pages/monitor.page";
import { v4 as uuidv4 } from "uuid";
import * as fs from "fs/promises";
import path from "path";

test.describe("Tutorial", () => {
  let buildPage: BuildPage;
  let monitorPage: MonitorPage;

  test.beforeEach(async ({ page, loginPage, testUser }, testInfo: TestInfo) => {
    buildPage = new BuildPage(page);
    monitorPage = new MonitorPage(page);

    // Start each test with login using worker auth
    await page.goto("/login");
    await loginPage.login(testUser.email, testUser.password);
    await test.expect(page).toHaveURL("/");
  });

  // test.afterAll(async ({}) => {
  //   // clear out the downloads folder
  //   console.log(
  //     `clearing out the downloads folder ${monitorPage.downloadsFolder}`,
  //   );

  //   await fs.rm(`${monitorPage.downloadsFolder}/monitor`, {
  //     recursive: true,
  //     force: true,
  //   });
  // });

  test("User can follow the tutorial", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: "Tutorial" }).click();
    // Welcom to Tutorial
    await buildPage.nextTutorialStep();
    // Open Blocks Menu
    await buildPage.openBlocksPanel();
    const calculatorBlock = await buildPage.getCalculatorBlockDetails();
    await buildPage.addBlock(calculatorBlock);

    // New Block Dialog
    await buildPage.nextTutorialStep();
    // Input to the Block
    await buildPage.nextTutorialStep();
    // Output from the Block
    await buildPage.nextTutorialStep();
    // Select Operation and Input Numbers
    await buildPage.selectBlockInputValue(
      calculatorBlock.id,
      "Operation",
      "Subtract",
    );
    await buildPage.fillBlockInputByPlaceholder(
      calculatorBlock.id,
      "For example: 10",
      "10",
    );
    await buildPage.fillBlockInputByPlaceholder(
      calculatorBlock.id,
      "For example: 5",
      "3",
    );
    // Press Save
    await buildPage.nextTutorialStep();
    await buildPage.saveAgent("Test Tutorial Agent");
    await buildPage.waitForSaveDialogClose();
    await buildPage.waitForRunTutorialButton();
    await buildPage.runAgent();

    // await buildPage.saveAgent("Test Tutorial Agent");
    // await buildPage.waitForVersionField();
    // await buildPage.runAgent();

    // Check the Ouput
    await buildPage.nextTutorialStep();

    await page
      .locator('[data-blockid="b1ab9b19-67a6-406d-abf5-2dba76d00c79"]')
      .click();
    await page
      .locator('[data-blockid="b1ab9b19-67a6-406d-abf5-2dba76d00c79"]')
      .press("ControlOrMeta+c");

    await page.locator("body").press("ControlOrMeta+v");

    // Get the other block (this is a bit of a hack, sorry)
    const dataId1 = await page
      .locator(".react-flow__node")
      .nth(0)
      .getAttribute("data-id");
    const dataId2 = await page
      .locator(".react-flow__node")
      .nth(1)
      .getAttribute("data-id");

    expect(dataId1).not.toBeNull();
    expect(dataId2).not.toBeNull();

    await buildPage.moveBlockToSide(dataId2 as string, "right", 300);


    // Move the block to the side
    await buildPage.nextTutorialStep();

    await buildPage.connectBlockOutputToBlockInputViaName(
      calculatorBlock.id,
      "Result",
      calculatorBlock.id,
      "A",
      dataId1 as string,
      dataId2 as string,
      true,
    );
    
    await buildPage.runAgent();
    await buildPage.nextTutorialStep();
    await buildPage.runAgent();
    // Congratulations
    await buildPage.nextTutorialStep();
    expect(await buildPage.isThereTutorialDialog()).toBe(false);
  });
});

