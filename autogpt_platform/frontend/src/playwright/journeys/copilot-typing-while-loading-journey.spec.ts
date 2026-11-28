import type { Locator, Page } from "@playwright/test";
import { currentSessionID, isChatStream, savedMessages } from "./copilot-chat";
import { expect, test } from "./fixtures";
import quickAnswerScript from "./model-scripts/quick-answer.json";
import {
  expectEachTurnOnce,
  type ModelScript,
  readJournal,
  requireScriptedModel,
  scriptedRun,
} from "./scripted-model";

const SCRIPT: ModelScript = quickAnswerScript;
const [ANSWER] = SCRIPT.turns.map((turn) => turn.text ?? "");
const PENDING_STATUS = "Your message will send as soon as your chat has loaded";

// Every script is held, so the first keys land in the server's HTML before any
// JavaScript runs; then the scripts load while typing goes on.
test.describe("typing into /home's message box while the page loads", () => {
  test("text and caret typed before the page's scripts run carry into the chat", async ({
    page,
    journeyUser: _user,
  }) => {
    const releaseScripts = await holdScripts(page);
    await page.goto("/home", { waitUntil: "commit" });

    const frame = page.getByTestId("chat-frame");
    await page.locator("#chat-frame-input").click();
    await page.keyboard.type("hello world");
    for (let i = 0; i < "world".length; i++) {
      await page.keyboard.press("ArrowLeft");
    }
    await page.keyboard.type("big ");
    await expect(frame).not.toHaveAttribute("data-hydrated");
    expect(await boxState(page.locator("#chat-frame-input"))).toEqual({
      value: "hello big world",
      caret: "hello big ".length,
      focused: true,
    });

    releaseScripts();
    await page.keyboard.type("brave ", { delay: 150 });
    await expect(frame).toHaveCount(0, { timeout: 60_000 });
    const chatBox = page.locator("#chat-input-empty");
    await expect(chatBox).toBeFocused();
    await page.keyboard.type("new ");

    expect(await boxState(chatBox)).toEqual({
      value: "hello big brave new world",
      caret: "hello big brave new ".length,
      focused: true,
    });
  });

  test("a message sent before the page's scripts run is sent once, when the chat takes it", async ({
    page,
    journeyUser: _user,
  }) => {
    test.setTimeout(150_000);
    await requireScriptedModel();
    const run = scriptedRun("quick-answer");
    const message = `Plan my week. ${run.marker}`;
    const sends: string[] = [];
    page.on("request", (request) => {
      if (isChatStream("POST")(request)) sends.push(request.url());
    });
    const releaseScripts = await holdScripts(page);
    await page.goto("/home", { waitUntil: "commit" });

    const frame = page.getByTestId("chat-frame");
    const frameBox = page.locator("#chat-frame-input");
    await frameBox.fill(message);
    await frameBox.press("Enter");
    await expect(frame).toHaveAttribute("data-send-pending", "");
    await expect(frame.getByRole("status")).toHaveText(PENDING_STATUS);
    await expect(frameBox, "the message waits in the box").toHaveValue(message);

    releaseScripts();
    await expect(page.getByText(ANSWER)).toBeVisible({ timeout: 90_000 });

    const sessionID = await currentSessionID(page);
    const asked = (await savedMessages(page, sessionID)).filter(
      (saved) => saved.role === "user",
    );
    expect(asked, "the chat should hold the message once").toHaveLength(1);
    expect(asked[0].content).toContain(message);
    expect(sends, "the chat should send it once").toHaveLength(1);
    expectEachTurnOnce(await readJournal(run.nonce), SCRIPT);
  });
});

/** Holds every script the page asks for until the returned call. */
async function holdScripts(page: Page) {
  let release = () => {};
  const released = new Promise<void>((resolve) => {
    release = resolve;
  });
  await page.route(/\/_next\/static\/.+\.js(\?.*)?$/, async (route) => {
    await released;
    await route.continue().catch(() => undefined);
  });
  return release;
}

function boxState(box: Locator) {
  return box.evaluate((element) => {
    const input = element as HTMLTextAreaElement;
    return {
      value: input.value,
      caret: input.selectionStart,
      focused: document.activeElement === input,
    };
  });
}
