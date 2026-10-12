import type { Page } from "@playwright/test";
import { expect, test } from "./fixtures";
import boardTaskScript from "./model-scripts/board-task.json";
import {
  type ModelScript,
  requireScriptedModel,
  scriptedRun,
} from "./scripted-model";

const SCRIPT: ModelScript = boardTaskScript;
const ANSWER = SCRIPT.turns[0].text ?? "";
const TITLE = "Write a haiku about boards";

test.use({ flagOverrides: { "task-board": true } });

test.describe("task board journey", () => {
  test.beforeAll(requireScriptedModel);

  test("a task added to Backlog starts in In progress with its chat, and Mark done moves it to Done", async ({
    page,
    journeyUser: _user,
  }) => {
    const run = scriptedRun("board-task");
    await page.goto("/home");

    // An empty board opens Backlog's input on its own.
    const backlog = column(page, "Backlog");
    const newTask = backlog.getByRole("textbox", { name: "New task" });
    await backlog
      .getByRole("button", { name: "Add a task…" })
      .or(newTask)
      .click();
    await newTask.fill(TITLE);
    await newTask.press("Enter");

    const backlogCard = backlog.getByRole("article", { name: TITLE });
    await expect(
      backlogCard.getByText("Not started"),
      "a new task should wait in Backlog until it starts",
    ).toBeVisible();

    await backlogCard.getByRole("button", { name: TITLE }).click();
    const panel = page.getByRole("dialog", { name: TITLE });
    await panel
      .getByRole("textbox", { name: "First message" })
      .fill(`${TITLE} ${run.marker}`);
    await panel.getByRole("button", { name: "Start" }).click();

    const chat = panel.getByRole("log");
    await expect(
      chat.getByText(TITLE),
      "the chat should show the first message",
    ).toBeVisible();
    await expect(
      chat.getByText(ANSWER, { exact: true }),
      "Otto should answer the first message",
    ).toBeVisible();
    await closePanel(page);

    const inProgressCard = column(page, "In progress").getByRole("article", {
      name: TITLE,
    });
    await expect(
      inProgressCard,
      "a started task should move to In progress",
    ).toBeVisible();

    await inProgressCard.getByRole("button", { name: TITLE }).click();
    const markDone = panel.getByRole("button", { name: "Mark done" });
    await markDone.click();
    await expect(markDone).toBeHidden();
    await closePanel(page);

    await expect(
      column(page, "Done").getByRole("article", { name: TITLE }),
      "Mark done should move the task to Done",
    ).toBeVisible();
    await expect(inProgressCard).toHaveCount(0);
  });
});

function column(page: Page, name: string) {
  return page.getByRole("region", { name, exact: true });
}

// The open panel hides the board from the accessibility tree.
async function closePanel(page: Page) {
  const panel = page.getByRole("dialog", { name: TITLE });
  await panel.getByRole("button", { name: "Close", exact: true }).click();
  await expect(panel).toBeHidden();
}
