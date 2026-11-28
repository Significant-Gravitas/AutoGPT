import { randomBytes } from "crypto";
import { expect, test as journeyTest } from "./fixtures";
import {
  createWorkflow,
  deleteSchedule,
  deleteScheduleInLibrary,
  finishOttosGreeting,
  groupsListing,
  issue,
  listSchedules,
  openHome,
  runRow,
  scheduleInLibrary,
  statusInLibrary,
  statusOnHome,
  UNNAMED_WORKFLOW,
  waitForHomeToList,
  type Workflow,
  workflowGroup,
} from "./home-recent-work";
import { pinKnownFailure } from "./known-failure";
import {
  observe,
  openRunDialog,
  rerunTask,
  startTask,
  waitForRunToEnd,
} from "./typed-inputs";

// Home's Recent work card as a person reads it: a group for each workflow,
// called what the Library calls it, holding its runs with the status the
// Library shows them in, and each schedule set up for it for as long as the
// schedule exists. The card's unit tests compose it from fixtures; this runs
// the real stack: the Library, the executor, the scheduler and the activity
// log the scheduler writes.

const test = journeyTest.extend<{ scheduledWorkflows: Workflow[] }>({
  // A schedule left behind keeps firing on a long-lived stack. A test adds
  // its workflow before scheduling it, so every schedule it has is deleted
  // even when the test stops before it learns the schedule's id.
  scheduledWorkflows: async ({ page }, provide) => {
    const workflows: Workflow[] = [];
    await provide(workflows);
    for (const workflow of workflows) {
      for (const schedule of await listSchedules(page, workflow)) {
        await deleteSchedule(page, schedule.id);
      }
    }
  },
});

test.describe("Home recent work journey", () => {
  // Someone who has chatted with Otto once: before that, Home shows Otto's
  // greeting instead of its cards.
  test.beforeEach(async ({ page, journeyUser: _user }) => {
    await finishOttosGreeting(page);
  });

  test(
    "two runs started in the Library are listed on Home in one group with the workflow's name, each with the status the Library shows",
    { annotation: issue(17357) },
    async ({ page, journeyUser: _user }, testInfo) => {
      const workflow = await createWorkflow(page, "Journey recent work");
      const first = await startTask(
        page,
        workflow,
        await openRunDialog(page, workflow),
      );
      const firstRun = await waitForRunToEnd(page, workflow, first.runID);
      expect(firstRun.status, "the first run should complete").toBe(
        "COMPLETED",
      );
      const second = await rerunTask(page, workflow);
      const secondRun = await waitForRunToEnd(page, workflow, second.runID);
      expect(secondRun.status, "the second run should complete").toBe(
        "COMPLETED",
      );
      const runIDs = [first.runID, second.runID];

      await page.reload();
      await expect(
        page.getByText(workflow.name).first(),
        "the Library should call the workflow by its name",
      ).toBeVisible();
      const inLibrary: Record<string, string> = {};
      for (const runID of runIDs) {
        inLibrary[runID] = await statusInLibrary(page, runID);
      }
      expect(inLibrary, "the Library should show both runs completed").toEqual({
        [first.runID]: "completed",
        [second.runID]: "completed",
      });

      const card = await openHome(page);
      const group = workflowGroup(card, workflow);
      await expect(
        group,
        "Home should list the workflow's runs in one group",
      ).toHaveCount(1);
      await expect(group.getByText("2 runs", { exact: true })).toBeVisible();
      const onHome: Record<string, string> = {};
      for (const runID of runIDs) {
        await expect(
          runRow(group, runID),
          "the group should list each run once",
        ).toHaveCount(1);
        onHome[runID] = await statusOnHome(runRow(group, runID));
      }
      expect(
        onHome,
        "Home should show each run in the status the Library shows",
      ).toEqual(inLibrary);
      await expect(
        card.getByText("2 completed", { exact: true }),
      ).toBeVisible();

      const shown = await group.getAttribute("aria-label");
      observe(testInfo, `Home calls the group ${JSON.stringify(shown)}`);
      // Only "Workflow task" is the known bug: any other name, or none, fails
      // here rather than under the pin.
      expect(
        [workflow.name, UNNAMED_WORKFLOW],
        `Home should call the group "${workflow.name}" or, as #17357 does, "${UNNAMED_WORKFLOW}"`,
      ).toContain(shown);
      const pinned = pinKnownFailure(
        `#17357: Home calls a workflow saved in the builder "${UNNAMED_WORKFLOW}"`,
        "Home should call the group by the workflow's name",
      );
      expect(shown, pinned).toBe(workflow.name);
    },
  );

  test(
    "a schedule made in a workflow's Library run dialog is listed on Home under that workflow, not under Otto",
    { annotation: issue(17360) },
    async ({ page, journeyUser: _user, scheduledWorkflows }, testInfo) => {
      const workflow = await createWorkflow(page, "Journey recent work");
      scheduledWorkflows.push(workflow);
      const schedule = await scheduleInLibrary(
        page,
        workflow,
        `Journey schedule ${uniqueStamp()}`,
      );
      await waitForHomeToList(page, schedule);

      const card = await openHome(page);
      await expect(
        card.getByText(schedule.name, { exact: true }),
        "Home should list the schedule",
      ).toBeVisible();
      const listedBy = await groupsListing(card, workflow, schedule.name);
      observe(testInfo, `Home lists the schedule in ${listedBy.join(", ")}`);
      await deleteSchedule(page, schedule.id);
      expect(
        [["the workflow"], ["Otto"]],
        "Home should list the schedule once, under its workflow or, as #17360 does, under Otto",
      ).toContainEqual(listedBy);
      const pinned = pinKnownFailure(
        "#17360: Home credits a schedule made in the Library to Otto",
        "Home should list the schedule under its workflow",
      );
      expect(listedBy, pinned).toEqual(["the workflow"]);
    },
  );

  test(
    "a schedule deleted in the Library is no longer listed on Home",
    { annotation: issue(17360) },
    async ({ page, journeyUser: _user, scheduledWorkflows }) => {
      const workflow = await createWorkflow(page, "Journey recent work");
      scheduledWorkflows.push(workflow);
      const schedule = await scheduleInLibrary(
        page,
        workflow,
        `Journey schedule ${uniqueStamp()}`,
      );
      // Listed first, so that it is gone afterwards means something.
      await waitForHomeToList(page, schedule);
      const before = await openHome(page);
      await expect(
        before.getByText(schedule.name, { exact: true }),
        "Home should list the schedule before it is deleted",
      ).toBeVisible();

      await deleteScheduleInLibrary(page, workflow, schedule);

      const after = await openHome(page);
      const pinned = pinKnownFailure(
        "#17360: Home keeps listing a schedule after it is deleted",
        "Home should no longer list the deleted schedule",
      );
      await expect(
        after.getByText(schedule.name, { exact: true }),
        pinned,
      ).toHaveCount(0, { timeout: 5_000 });
    },
  );
});

function uniqueStamp() {
  return `${Date.now()}-${randomBytes(3).toString("hex")}`;
}
