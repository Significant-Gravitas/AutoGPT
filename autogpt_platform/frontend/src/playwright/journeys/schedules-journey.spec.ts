import { randomBytes } from "crypto";
import type { Page } from "@playwright/test";
import { expect, test } from "../coverage-fixture";
import { createUniqueAgentName } from "../pages/build.page";
import { LibraryPage } from "../pages/library.page";

const TIMEZONE = "America/New_York";
const STORE_VALUE_BLOCK_ID = "1ff065e9-88e8-4358-9d82-8dc91f622ba9";

test.use({ storageState: { cookies: [], origins: [] }, timezoneId: TIMEZONE });

interface Schedule {
  id: string;
  name: string;
  cron: string;
  next_run_time: string;
}

test.describe("schedules journey", () => {
  test("a weekday schedule can be created, edited, and deleted from the Library, and every change survives a reload", async ({
    page,
  }) => {
    test.setTimeout(90_000);
    await signUpJourneyUser(page);
    const agentName = createUniqueAgentName("Journey schedule");
    const graphId = await createAgent(page, agentName);

    await new LibraryPage(page).openSavedAgent(agentName);
    await page
      .getByRole("button", { name: /Setup your task|New agent task/i })
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

// Onboarding steps that dismiss the first-visit cards covering the Library.
const COMPLETED_ONBOARDING_STEPS = [
  "ONBOARDING_COMPLETE",
  "AGENT_NEW_RUN",
  "CAPABILITY_CARDS",
  "AGENTS_TAB_INTRO",
  "MARKETPLACE_TAB_INTRO",
  "BUILD_TAB_INTRO",
  "WORKFLOWS_MOVED",
];

// A brand-new user by default. Previews paywall fresh users, so there
// PLAYWRIGHT_JOURNEY_EMAIL and PLAYWRIGHT_JOURNEY_PASSWORD sign in as an
// existing subscribed account instead.
async function signUpJourneyUser(page: Page) {
  const email = process.env.PLAYWRIGHT_JOURNEY_EMAIL;
  const password = process.env.PLAYWRIGHT_JOURNEY_PASSWORD;
  if (email && password) {
    const signIn = await page.request.post("/api/auth/sign-in/email", {
      headers: { Origin: appOrigin() },
      data: { email, password },
    });
    expect(signIn.ok(), `sign-in failed: ${signIn.status()}`).toBe(true);
  } else {
    const id = `${Date.now()}-${randomBytes(3).toString("hex")}`;
    const signUp = await page.request.post("/api/auth/sign-up/email", {
      headers: { Origin: appOrigin() },
      data: {
        email: `journey-${id}@${journeyEmailDomain()}`,
        password: `Journey-${randomBytes(12).toString("hex")}`,
        name: `Journey ${id}`,
      },
    });
    expect(signUp.ok(), `sign-up failed: ${signUp.status()}`).toBe(true);
  }
  for (const step of COMPLETED_ONBOARDING_STEPS) {
    const onboarding = await page.request.post(
      "/api/proxy/api/onboarding/step",
      { params: { step } },
    );
    expect(onboarding.ok(), `onboarding step ${step} failed`).toBe(true);
  }
  const timezone = await page.request.post(
    "/api/proxy/api/auth/user/timezone",
    { data: { timezone: TIMEZONE } },
  );
  expect(timezone.ok(), "setting the user's timezone failed").toBe(true);
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

function appOrigin() {
  return new URL(process.env.PLAYWRIGHT_BASE_URL ?? "http://localhost:3000")
    .origin;
}

// Previews only accept sign-ups from allowlisted domains; @previews.agpt.co
// keeps password sign-up and needs no inbox.
function journeyEmailDomain() {
  if (process.env.PLAYWRIGHT_JOURNEY_EMAIL_DOMAIN) {
    return process.env.PLAYWRIGHT_JOURNEY_EMAIL_DOMAIN;
  }
  const { hostname } = new URL(appOrigin());
  return ["localhost", "127.0.0.1", "[::1]"].includes(hostname)
    ? "example.com"
    : "previews.agpt.co";
}
