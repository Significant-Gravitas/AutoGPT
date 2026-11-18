import type { Page } from "@playwright/test";
import { createUniqueAgentName } from "../pages/build.page";
import { LibraryPage } from "../pages/library.page";
import { expect, test } from "./fixtures";

const TIMEZONE = "America/New_York";
const STORE_VALUE_BLOCK_ID = "1ff065e9-88e8-4358-9d82-8dc91f622ba9";
const SCHEDULES_UNAVAILABLE =
  "Schedules are temporarily unavailable — your schedules are safe and will run once it's back.";
// The schedule lists a restarting scheduler answers 503.
const SCHEDULE_READS =
  /\/api\/proxy\/api\/(schedules(\/followups)?|graphs\/[^/]+\/schedules)(\?.*)?$/;
// Marks the 503s this journey fakes, so a real one still fails it.
const INJECTED = { name: "x-journey-injected", value: "scheduler-outage" };

test.use({
  timezoneId: TIMEZONE,
  expectedResponses: [
    { status: 503, method: "GET", url: SCHEDULE_READS, header: INJECTED },
  ],
});

interface Schedule {
  id: string;
  name: string;
  cron: string;
  next_run_time: string;
}

test.describe("schedules journey", () => {
  test("a weekday schedule can be created, edited, and deleted from the Library, and every change survives a reload", async ({
    page,
    journeyUser: _user,
  }) => {
    await setTimezone(page);
    const agentName = createUniqueAgentName("Journey schedule");
    const graphId = await createAgent(page, agentName);

    await new LibraryPage(page).openSavedAgent(agentName);
    await page
      .getByRole("button", { name: /Setup your task|New workflow task/i })
      .first()
      .click();
    await page.getByRole("button", { name: "Schedule Task" }).click();
    const createDialog = page.getByRole("dialog", { name: "Schedule Run" });
    await createDialog.getByLabel("Schedule Name").fill("Weekday report");
    await selectOption(page, createDialog.locator("#repeat"), "Weekly");
    await createDialog.getByRole("button", { name: "Weekdays" }).click();
    await createDialog.getByRole("button", { name: "Schedule" }).click();
    // Saving waits on the scheduler service, which is slow on a cold stack.
    await expect(createDialog).toBeHidden({ timeout: 30_000 });

    const created = await onlySchedule(page, graphId);
    expect(created.cron).toBe("0 9 * * 1,2,3,4,5");
    expectWeekdayAt(created, 9, 0);
    await openSchedule(page, "Weekday report");
    await expect(nextRunText(page)).toContainText("09:00");

    await page.getByRole("button", { name: "Edit schedule" }).first().click();
    const editDialog = page.getByRole("dialog", { name: "Edit Schedule" });
    await editDialog.locator("#schedule-name").fill("Evening report");
    await editDialog.locator("#schedule-time").fill("18:30");
    await editDialog.getByRole("button", { name: "Save" }).click();
    await expect(editDialog).toBeHidden({ timeout: 30_000 });

    await page.reload();
    await openSchedule(page, "Evening report");
    await expect(nextRunText(page)).toContainText("18:30");
    const edited = await onlySchedule(page, graphId);
    expect(edited.id).toBe(created.id);
    expect(edited.cron).toBe("30 18 * * 1,2,3,4,5");
    expectWeekdayAt(edited, 18, 30);

    // The scheduler restarts on every deploy. Meanwhile the pages say its
    // schedules are unavailable instead of failing or listing none, and they
    // list them again once it answers, without a reload.
    const agentPage = page.url();
    await whileSchedulerIsOut(page, async () => {
      await page.goto("/library/followups");
      await expect(page.getByText(SCHEDULES_UNAVAILABLE)).toBeVisible();
      await expect(page.getByTestId("followups-empty")).toHaveCount(0);
      await page.goto(agentPage);
      await expect(page.getByText(SCHEDULES_UNAVAILABLE).first()).toBeVisible();
      await expect(page.getByText("Something went wrong")).toHaveCount(0);
    });
    // Try again, or the recheck after Retry-After if it lands first.
    const tryAgain = page.getByRole("button", {
      name: "Try again",
      exact: true,
    });
    await expect(async () => {
      if (await tryAgain.isVisible()) await tryAgain.click({ timeout: 1_000 });
      await expect(page.getByText(SCHEDULES_UNAVAILABLE)).toHaveCount(0, {
        timeout: 1_000,
      });
    }).toPass({ timeout: 20_000 });
    await openSchedule(page, "Evening report");

    await page.getByRole("button", { name: "Delete schedule" }).first().click();
    await page.getByRole("button", { name: "Delete Schedule" }).click();
    await expect.poll(() => listSchedules(page, graphId)).toEqual([]);
    await page.reload();
    await expect(page.getByText("Evening report")).toHaveCount(0);
    const runs = await page.request.get(
      `/api/proxy/api/graphs/${graphId}/executions`,
    );
    expect(await runs.json()).toMatchObject({ executions: [] });
  });
});

async function whileSchedulerIsOut(page: Page, check: () => Promise<void>) {
  await page.route(SCHEDULE_READS, (route) =>
    route.request().method() === "GET"
      ? route.fulfill({
          status: 503,
          headers: { "Retry-After": "15", [INJECTED.name]: INJECTED.value },
          contentType: "application/json",
          body: JSON.stringify({
            message: "Failed to process GET /api/schedules",
            detail:
              "Schedules are temporarily unavailable. Try again in a moment.",
            hint: "Retry in 15 seconds.",
          }),
        })
      : route.fallback(),
  );
  try {
    await check();
  } finally {
    await page.unroute(SCHEDULE_READS);
  }
}

function nextRunText(page: Page) {
  return page.getByText("Next run", { exact: true }).locator("xpath=..");
}

async function openSchedule(page: Page, name: string) {
  await page.getByRole("tab", { name: /^Scheduled/ }).click();
  await page.getByText(name, { exact: true }).first().click();
  await expect(page.getByText("Next run", { exact: true })).toBeVisible();
}

async function selectOption(
  page: Page,
  trigger: ReturnType<Page["locator"]>,
  option: string,
) {
  await trigger.click();
  await page.getByRole("option", { name: option, exact: true }).click();
}

async function listSchedules(page: Page, graphId: string) {
  const response = await page.request.get(
    `/api/proxy/api/graphs/${graphId}/schedules`,
  );
  expect(response.ok(), `listing schedules: ${response.status()}`).toBe(true);
  return (await response.json()) as Schedule[];
}

async function onlySchedule(page: Page, graphId: string) {
  const schedules = await listSchedules(page, graphId);
  expect(schedules).toHaveLength(1);
  return schedules[0];
}

// The scheduler must evaluate the cron in the user's timezone: the next run
// is a Monday-to-Friday date at exactly the wall-clock time they picked.
function expectWeekdayAt(schedule: Schedule, hour: number, minute: number) {
  const parts = Object.fromEntries(
    new Intl.DateTimeFormat("en-US", {
      timeZone: TIMEZONE,
      weekday: "short",
      hour: "2-digit",
      minute: "2-digit",
      hourCycle: "h23",
    })
      .formatToParts(new Date(schedule.next_run_time))
      .map((p) => [p.type, p.value]),
  );
  expect(["Mon", "Tue", "Wed", "Thu", "Fri"]).toContain(parts.weekday);
  expect([Number(parts.hour), Number(parts.minute)]).toEqual([hour, minute]);
}

async function setTimezone(page: Page) {
  const response = await page.request.post(
    "/api/proxy/api/auth/user/timezone",
    { data: { timezone: TIMEZONE } },
  );
  expect(response.ok(), "setting the user's timezone failed").toBe(true);
}

async function createAgent(page: Page, name: string): Promise<string> {
  const response = await page.request.post("/api/proxy/api/graphs", {
    data: {
      source: "upload",
      graph: {
        name,
        description: "Stores a value, for the schedules journey",
        is_active: true,
        nodes: [
          {
            id: "store",
            block_id: STORE_VALUE_BLOCK_ID,
            input_default: { input: "scheduled" },
            metadata: { position: { x: 0, y: 0 } },
            input_links: [],
            output_links: [],
          },
        ],
        links: [],
      },
    },
  });
  expect(response.ok(), `creating the agent failed: ${response.status()}`).toBe(
    true,
  );
  return (await response.json()).id;
}
