import { openLongSession } from "../lib/copilot-page";
import { settle, type Scenario } from "../lib/scenario";

const MOVES = 60;
const DRAG_MS = 1_000;
// 60 moves of 2 px widen the 432 px default panel to 552 px, under the 600 px
// cap, so every move asks for a new width.
const STEP_PX = 2;
const READY_TIMEOUT_MS = 30_000;

export const copilotResizeDrag: Scenario = {
  name: "copilot-resize-drag",
  description:
    "Open the files panel of a 200-message chat and drag its resize handle 120 px wider over 1 s (60 pointer moves).",
  readyWhen:
    "the newest user message and the composer are visible, the files panel is open, then network idle + 1 s",
  windowMs: 1_000,
  async setup(ctx) {
    const { page } = ctx;
    await openLongSession({ ctx });
    await page.getByRole("button", { name: /^Open files/ }).click();
    const handle = page.locator("[data-context-panel] [role='separator']");
    await handle.waitFor({ state: "visible", timeout: READY_TIMEOUT_MS });
    await settle(page);
    const box = await handle.boundingBox();
    if (!box)
      throw new Error("Context panel resize handle has no bounding box");
    return {
      async run() {
        const x = box.x + box.width / 2;
        const y = box.y + box.height / 2;
        await page.mouse.move(x, y);
        await page.mouse.down();
        for (let move = 1; move <= MOVES; move++) {
          await page.mouse.move(x - move * STEP_PX, y);
          await page.waitForTimeout(DRAG_MS / MOVES);
        }
        await page.mouse.up();
      },
    };
  },
};
