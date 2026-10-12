import type { Page } from "@playwright/test";
import { expect, test } from "./fixtures";
import boardTaskScript from "./model-scripts/board-task.json";
import {
  expectChargedWhatWasBilled,
  expectEachTurnOnce,
  type ModelScript,
  readJournal,
  requireScriptedModel,
  scriptedRun,
} from "./scripted-model";

const SCRIPT: ModelScript = boardTaskScript;
const ANSWER = SCRIPT.turns[0].text ?? "";
const TITLE = "Write a haiku about boards";
const PROJECT = "Launch";
const PROJECT_TASK = "Draft the launch post";

test.use({ flagOverrides: { "task-board": true } });

test.describe("task board journey", () => {
  test.beforeAll(requireScriptedModel);

  test("a task added to Backlog starts in In progress with its chat, Mark done moves it to Done, and a new project's task moves to Home", async ({
    page,
    journeyUser,
  }) => {
    const run = scriptedRun("board-task");
    const firstMessage = `${TITLE} ${run.marker}`;
    await page.goto("/home");

    const backlogCard = await addTask(page, TITLE);
    await expect(
      backlogCard.getByText("Not started"),
      "a new task should wait in Backlog until it starts",
    ).toBeVisible();

    await backlogCard.getByRole("button", { name: TITLE }).click();
    const panel = page.getByRole("dialog", { name: TITLE });
    await panel
      .getByRole("textbox", { name: "First message" })
      .fill(firstMessage);
    await panel.getByRole("button", { name: "Start" }).click();

    const chat = panel.getByRole("log");
    await expect(
      chat.getByText(firstMessage, { exact: true }),
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

    const journal = await readJournal(run.nonce);
    expectEachTurnOnce(journal, SCRIPT);
    if (!journeyUser.id) throw new Error("the journey user should have an id");
    await expectChargedWhatWasBilled(journeyUser.id, journal);

    await page.getByRole("button", { name: "New project" }).click();
    await expect(page).toHaveURL(/\/boards\//);
    const projectName = page.getByRole("textbox", { name: "Project name" });
    await expect(
      projectName,
      "a new project should open with its name ready to type over",
    ).toHaveValue("Untitled project");
    await projectName.fill(PROJECT);
    await projectName.press("Enter");
    await expect(
      page.getByRole("heading", { name: PROJECT, level: 1 }),
      "Enter should save the project's name",
    ).toBeVisible();
    await expect(
      page.getByRole("link", { name: PROJECT, exact: true }),
      "the sidebar should list the renamed project",
    ).toBeVisible();

    const projectCard = await addTask(page, PROJECT_TASK);
    await projectCard.getByRole("button", { name: "Task actions" }).click();
    await page.getByRole("menuitem", { name: "Move to project" }).click();
    await page.getByRole("menuitem", { name: "Home", exact: true }).click();
    await expect(
      page
        .getByRole("region", { name: /^Notifications/ })
        .getByText("Moved to Home"),
      "moving a task should say where it went",
    ).toBeVisible();
    await expect(
      projectCard,
      "a moved task should leave the project",
    ).toHaveCount(0);

    await page.getByRole("link", { name: "Home", exact: true }).click();
    await expect(page).toHaveURL(/\/home/);
    await expect(
      column(page, "Backlog").getByRole("article", { name: PROJECT_TASK }),
      "a task moved to Home should keep its Backlog column",
    ).toBeVisible();
  });
});

// An empty board opens Backlog's input on its own.
async function addTask(page: Page, title: string) {
  const backlog = column(page, "Backlog");
  const newTask = backlog.getByRole("textbox", { name: "New task" });
  await backlog
    .getByRole("button", { name: "Add a task…" })
    .or(newTask)
    .click();
  await newTask.fill(title);
  await newTask.press("Enter");
  return backlog.getByRole("article", { name: title });
}

function column(page: Page, name: string) {
  return page.getByRole("region", { name, exact: true });
}

// The open panel hides the board from the accessibility tree.
async function closePanel(page: Page) {
  const panel = page.getByRole("dialog", { name: TITLE });
  await panel.getByRole("button", { name: "Close", exact: true }).click();
  await expect(panel).toBeHidden();
}
