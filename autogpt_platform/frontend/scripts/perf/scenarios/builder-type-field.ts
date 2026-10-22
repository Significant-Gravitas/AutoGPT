import { BuildPage } from "../../../src/playwright/pages/build.page";
import { createBuilderFixture } from "../lib/builder-fixture";
import { installRoutes } from "../lib/routes";
import { settle, type Scenario } from "../lib/scenario";

const TEXT = "twenty chars of text";
const KEY_DELAY_MS = 50;
const READY_TIMEOUT_MS = 60_000;

export const builderTypeField: Scenario = {
  name: "builder-type-field",
  description: `Type ${TEXT.length} characters, ${KEY_DELAY_MS} ms apart, into a text field of the first node in an idle 50-node workflow.`,
  readyWhen:
    "all 50 nodes are on the canvas and the first node's text field is focused, then network idle + 1 s",
  windowMs: 2_000,
  async setup(ctx) {
    const { page } = ctx;
    const fixture = createBuilderFixture(ctx.userID);
    await installRoutes(page, fixture.handlers);
    const builder = new BuildPage(page);
    await page.goto(`/build?flowID=${fixture.graph.id}`);
    await builder.closeTutorial();
    const nodes = page.locator('[data-id^="custom-node-"]');
    await nodes
      .nth(fixture.graph.nodes.length - 1)
      .waitFor({ state: "attached", timeout: READY_TIMEOUT_MS });
    const field = nodes.first().locator("input:visible, textarea:visible");
    await field
      .first()
      .waitFor({ state: "visible", timeout: READY_TIMEOUT_MS });
    await field.first().click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Backspace");
    await settle(page);
    return {
      async run() {
        await page.keyboard.type(TEXT, { delay: KEY_DELAY_MS });
      },
    };
  },
};
