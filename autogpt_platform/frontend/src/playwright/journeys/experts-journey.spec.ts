import type { Locator, Page } from "@playwright/test";
import {
  currentSessionID,
  expectChatSettled,
  firstWords,
  isChatStream,
  savedMessages,
} from "./copilot-chat";
import { fakeAPIKey, observe, uniqueStamp } from "./credentials";
import {
  createOwnWorkflow,
  deleteSchedules,
  type Expert,
  expectQuestion,
  hireThroughAPI,
  installOnExpert,
  isPath,
  listExpertChats,
  listHiredExperts,
  listSchedules,
  listSetupItems,
  listTemplates,
  MODEL_TURN_TIMEOUT,
  onboardingCard,
  pendingOnboarding,
  pickOption,
  type Schedule,
  scheduledExpert,
  scheduledWorkflow,
  scheduleStates,
  scheduleThroughAPI,
  type SetupItem,
  teamCard,
  waitForSetup,
} from "./experts";
import { expect, test } from "./fixtures";
import kickoffScript from "./model-scripts/expert-kickoff.json";
import pauseScript from "./model-scripts/expert-pause-schedule.json";
import quickScript from "./model-scripts/quick-answer.json";
import {
  expectEachTurnOnce,
  type ModelScript,
  readJournal,
  releaseHeldReply,
  requireScriptedModel,
  scriptedRun,
} from "./scripted-model";
import { JOURNEY_USER_AGENT } from "./user-agent";

// Hire an expert from the Marketplace, work through its kickoff, chat with it
// from Team, connect what it needs, then let it go and hire it again.
//
// The expert is the roster's one with a workflow on a cadence (see
// `scheduledExpert`): the roster comes from the skills catalog, and its
// workflows are official listings the journeys leg loads before publishing it.

const KICKOFF: ModelScript = kickoffScript;
const PAUSE: ModelScript = pauseScript;
const QUICK: ModelScript = quickScript;

interface Step {
  question: string;
  options: string[];
}
// The kickoff card's two questions, as its script asks them.
const [FIRST, SECOND] = KICKOFF.turns[0].tool_calls?.[0].input.steps as Step[];
// The rest of the kickoff turn, which streams until the journey releases it.
const HELD_TEXT = KICKOFF.turns[1].text ?? "";
const ANSWERS =
  `**Here are my answers:**\n\n> ${FIRST.question}\n\n${FIRST.options[0]}` +
  `\n\n> ${SECOND.question}\n\n${SECOND.options[0]}\n\nPlease proceed.`;

const PAUSED_SCHEDULE = "Paused before letting go";
const RUNNING_SCHEDULE = "Kept running";

// Three model turns (the kickoff's two requests, the answers, a fresh chat).
const ONBOARDING_TEST_TIMEOUT = 180_000;
// One chat of three model requests, two hires and a schedule dialog.
const REHIRE_TEST_TIMEOUT = 150_000;
// Three waits on background setup (up to a minute each) and some navigation,
// no model turns.
const SETUP_TEST_TIMEOUT = 210_000;

// Held replies a test started but didn't get to release: each release lets
// one held reply through, so only a failed test's are released after it,
// and the scripted model isn't left holding a turn open.
const unreleased = new Set<string>();

test.describe("experts journey", () => {
  test.beforeAll(requireScriptedModel);
  test.afterEach(async () => {
    for (const nonce of unreleased) {
      await releaseHeldReply(nonce).catch(() => undefined);
    }
    unreleased.clear();
  });

  test("an expert found by its job title is hired, and its kickoff card is answered mid-turn and from Team", async ({
    page,
    journeyUser: _user,
    browser,
    baseURL,
  }) => {
    test.setTimeout(ONBOARDING_TEST_TIMEOUT);
    const expert = scheduledExpert(await listTemplates(page));
    const jobTitle = expert.job_title as string;

    await test.step("its public profile names it in the server's HTML", async () => {
      // What a crawler reads without running a script: the page as the
      // server sent it, with scripts switched off.
      const crawler = await browser.newContext({
        baseURL,
        javaScriptEnabled: false,
        storageState: { cookies: [], origins: [] },
        userAgent: JOURNEY_USER_AGENT,
      });
      try {
        const visitor = await crawler.newPage();
        const response = await visitor.goto(
          `/marketplace/experts/${expert.id}`,
          { waitUntil: "domcontentloaded" },
        );
        expect(response?.status(), "a signed-out visitor gets the page").toBe(
          200,
        );
        expect(await visitor.title()).toContain(`${expert.name}, ${jobTitle}`);
        expect(
          await visitor.locator("h1").allTextContents(),
          "the page should be headed with the expert's name",
        ).toContain(expert.name);
        await expect(
          visitor.getByText(jobTitle, { exact: true }).first(),
          "the page should show the expert's job title",
        ).toBeAttached();
      } finally {
        await crawler.close();
      }
    });

    await test.step("Marketplace search finds it by its job title", async () => {
      await page.goto(
        `/marketplace/search?searchTerm=${encodeURIComponent(jobTitle)}`,
      );
      const card = page
        .getByRole("region", { name: "Experts" })
        .locator(`a[href="/marketplace/experts/${expert.id}"]`);
      await expect(card).toBeVisible();
      await card.click();
      await page.waitForURL(`**/marketplace/experts/${expert.id}`);
    });

    const kickoff = scriptedRun("expert-kickoff");
    unreleased.add(kickoff.nonce);
    let hiredID = "";
    await test.step("signed in, its page offers Hire, and hiring opens its first chat", async () => {
      const hire = page.getByRole("button", {
        name: `Hire ${expert.name}`,
        exact: true,
      });
      await expect(hire).toBeVisible();
      await expect(page.getByText("Coming soon")).toHaveCount(0);
      const hired = page.waitForResponse(
        isPath("POST", "/api/proxy/api/experts"),
      );
      await hire.click();
      const response = await hired;
      expect(response.ok(), `hiring: ${response.status()}`).toBe(true);
      hiredID = ((await response.json()) as { expert: Expert }).expert.id;

      // The platform writes the kickoff message itself, so the script's
      // marker rides in the writing sample, which the system prompt carries.
      const picker = page.getByRole("dialog");
      await expect(
        picker.getByRole("heading", {
          name: `How should ${expert.name} write?`,
        }),
      ).toBeVisible();
      await picker
        .getByLabel("Custom voice sample")
        .fill(`Short, plain sentences. ${kickoff.marker}`);
      const voiceSaved = page.waitForResponse(
        isPath("PATCH", `/api/proxy/api/experts/${hiredID}/soul`),
      );
      await picker.getByRole("button", { name: "Use this voice" }).click();
      expect((await voiceSaved).ok(), "saving the voice").toBe(true);
      await page.waitForURL(
        (url) =>
          url.pathname === "/copilot" &&
          url.searchParams.get("expertId") === hiredID,
      );
    });

    const transcript = page.getByRole("log");
    const card = onboardingCard(transcript);
    let sessionID = "";
    await test.step("the kickoff's card takes an answer while the turn still streams", async () => {
      await expect(
        card.getByRole("progressbar", { name: "Setup progress" }),
        "the kickoff should open the onboarding card",
      ).toBeVisible({ timeout: MODEL_TURN_TIMEOUT });
      await expectQuestion(card, "Question 1 of 2", FIRST.question);
      // The rest of the turn waits at its hold: the turn is still running.
      await expect
        .poll(async () => heldTurns(kickoff.nonce), {
          message: "the kickoff's second request should be held open",
        })
        .toEqual([
          [1, true],
          [2, null],
        ]);
      // The chat shows a word only once the next one starts, so the held
      // reply's first three words show as two.
      await expect(
        transcript.getByText(firstWords(HELD_TEXT, 2)),
      ).toBeVisible();
      sessionID = await currentSessionID(page);
      await pickOption(card, FIRST.options[0]);
      await expectQuestion(card, "Question 2 of 2", SECOND.question);
    });

    await test.step("once the turn ends, the card still waits on question 2", async () => {
      await releaseHeldReply(kickoff.nonce);
      unreleased.delete(kickoff.nonce);
      await expect(transcript.getByText(HELD_TEXT)).toBeVisible();
      await expect(
        page.getByText("Reply finished", { exact: true }),
        "the turn should end and settle",
      ).toBeAttached({ timeout: MODEL_TURN_TIMEOUT });
      await expectChatSettled(page, sessionID);
      // Settling re-keys the turn's rows, which remounts the card.
      await expectQuestion(card, "Question 2 of 2", SECOND.question);
      await expect(
        card.getByRole("radio", { name: SECOND.options[0], exact: true }),
      ).toBeEnabled();
      expect(await pendingOnboarding(page, hiredID)).toEqual({
        session_id: sessionID,
      });
    });

    const drawer = page.getByRole("complementary", {
      name: `Chat with ${expert.name}`,
    });
    const drawerCard = onboardingCard(drawer);
    await test.step("on Team, Chat opens its chat at the waiting card, and a new chat finds it again", async () => {
      await page.goto("/team");
      await teamCard(page, expert.name)
        .getByRole("button", { name: "Chat", exact: true })
        .click();
      await expectQuestion(drawerCard, "Question 2 of 2", SECOND.question);
      const lookedUp = page.waitForResponse(
        isPath("GET", `/api/proxy/api/experts/${hiredID}/onboarding`),
      );
      await drawer.getByRole("button", { name: "New task" }).click();
      await lookedUp;
      await expectQuestion(drawerCard, "Question 2 of 2", SECOND.question);
    });

    await test.step("answered from the drawer, the card goes out once and the expert carries on", async () => {
      await pickOption(drawerCard, SECOND.options[0]);
      await expect(
        drawerCard.getByRole("progressbar", { name: "Setup progress" }),
      ).toHaveAttribute("aria-valuetext", "Check your answers");
      const sent = page.waitForRequest(isChatStream("POST"));
      await drawerCard.getByRole("button", { name: "Send answers" }).click();
      await sent;
      await expect(drawer.getByText(KICKOFF.turns[2].text ?? "")).toBeVisible({
        timeout: MODEL_TURN_TIMEOUT,
      });
      await expect(
        drawer.getByText(`Setup questions from ${expert.name}`),
        "the card should settle into history",
      ).toBeVisible();
      await expectChatSettled(page, sessionID);
      const answers = (await savedMessages(page, sessionID)).filter(
        (message) =>
          message.role === "user" &&
          message.content?.includes("**Here are my answers:**"),
      );
      expect(answers.map((message) => message.content?.trim())).toEqual([
        ANSWERS,
      ]);
      expect(
        await pendingOnboarding(page, hiredID),
        "nothing should wait on the user any more",
      ).toBeNull();
    });

    await test.step("a new chat now opens fresh", async () => {
      const quick = scriptedRun("quick-answer");
      await drawer.getByRole("button", { name: "New task" }).click();
      await expect(drawer.getByText("What can I do for you?")).toBeVisible();
      await sendInChat(
        page,
        drawer,
        `${quick.marker} What will you take on first?`,
      );
      await expect(drawer.getByText(QUICK.turns[0].text ?? "")).toBeVisible({
        timeout: MODEL_TURN_TIMEOUT,
      });
      const chats = await listExpertChats(page, hiredID);
      expect(chats, "the fresh chat should be a second chat").toHaveLength(2);
      expect(chats).toContain(sessionID);
      expectEachTurnOnce(await readJournal(quick.nonce), QUICK);
    });

    expectEachTurnOnce(await readJournal(kickoff.nonce), KICKOFF);
  });

  test("connecting a service from Team's Setup needed reaches every expert list, and no schedule starts without its inputs", async ({
    page,
    journeyUser: _user,
  }, testInfo) => {
    test.setTimeout(SETUP_TEST_TIMEOUT);
    const templates = await listTemplates(page);
    const roster = scheduledExpert(templates);
    const teammateTemplate = templates.find((t) => t.id !== roster.id);
    if (!teammateTemplate)
      throw new Error("the roster should have two experts");
    const expert = await hireThroughAPI(page, roster.id);
    const teammate = await hireThroughAPI(page, teammateTemplate.id);
    const hired = await waitForSetup(page, expert.id);
    await waitForSetup(page, teammate.id);
    const workflow = scheduledWorkflow(hired);
    expect(
      workflow.schedule_id,
      "a workflow that needs inputs nobody gave should get no schedule",
    ).toBeNull();

    const items = await listSetupItems(page, expert.id);
    observe(
      testInfo,
      `setup items: ${JSON.stringify(items.map((i) => [i.resolution, i.providers, i.missing_inputs]))}`,
    );
    const inputs = items.find(
      (item) =>
        item.resolution === "inputs" && item.workflow_id === workflow.id,
    );
    expect(
      inputs?.missing_inputs.length,
      "Setup needed should ask for the inputs the cadence can't supply",
    ).toBeGreaterThan(0);
    const connect = await apiKeyItem(page, items);
    const provider = connect.providers[0];
    const keyName = `Experts journey ${uniqueStamp()}`;

    const section = page.getByTestId("expert-integrations-section");
    await test.step("each expert's Integrations start empty", async () => {
      await page.goto(`/team/${teammate.id}?tab=integrations`);
      await expect(section.getByText("No integrations yet")).toBeVisible();
      await expect(
        section.getByRole("button", { name: "Use existing" }).first(),
      ).toBeDisabled();
      // In-app from here on, so a list left stale would still show it.
      await backToTeam(page);
      await openIntegrations(page, expert.name);
      await expect(section.getByText("No integrations yet")).toBeVisible();
      await backToTeam(page);
    });

    const setup = page.getByRole("region", { name: "Setup needed" });
    // The expert's own row: its teammate may need the same service.
    const forExpert = new RegExp(` for ${escapeRegExp(expert.name)}$`);
    const row = setup
      .getByRole("article")
      .filter({ has: page.getByRole("img", { name: provider, exact: true }) })
      .filter({ has: page.getByText(forExpert) });
    let service = "";
    await test.step("Setup needed connects the service, and the expert's item clears", async () => {
      await expect(row).toHaveCount(1);
      const title = await row.getByText(forExpert).textContent();
      service = (title ?? "").replace(` for ${expert.name}`, "");
      await row.getByRole("button", { name: "Connect", exact: true }).click();
      const dialog = page.getByRole("dialog");
      await expect(dialog.getByText(/^Connect AutoGPT to /)).toBeVisible();
      await dialog.getByRole("button", { name: /API Key/ }).click();
      await dialog
        .getByRole("textbox", { name: "Name", exact: true })
        .fill(keyName);
      await dialog.getByLabel("API key", { exact: true }).fill(fakeAPIKey());
      const saved = page.waitForResponse(
        isPath("POST", `/api/proxy/api/integrations/${provider}/credentials`),
      );
      const granted = page.waitForResponse(
        isPath("POST", `/api/proxy/api/experts/${expert.id}/credentials`),
      );
      await dialog.getByRole("button", { name: "Continue" }).click();
      expect((await saved).ok(), "saving the key").toBe(true);
      expect((await granted).ok(), "granting it to the expert").toBe(true);
      await expect(dialog).toBeHidden();
      await expect(
        row,
        "the expert's item for the service should clear",
      ).toHaveCount(0);
      await expect(
        teamCard(page, expert.name)
          .getByRole("list", { name: "Integrations" })
          .getByRole("img", { name: service, exact: true }),
        "the expert's card should show the service",
      ).toBeVisible();
    });

    await test.step("the key reaches every expert list, without a reload", async () => {
      await openIntegrations(page, expert.name);
      await expect(
        section
          .getByTestId("expert-integration-row")
          .filter({ hasText: keyName }),
      ).toBeVisible();
      await backToTeam(page);
      await openIntegrations(page, teammate.name);
      await expect(
        section.getByText("No integrations yet"),
        "the teammate wasn't given it",
      ).toBeVisible();
      await section
        .getByRole("button", { name: "Use existing" })
        .first()
        .click();
      const useExisting = page.getByRole("dialog", {
        name: "Use an existing connection",
      });
      await expect(
        useExisting.getByText(keyName),
        "the teammate can be given the new key",
      ).toBeVisible();
      await page.keyboard.press("Escape");
      await expect(useExisting).toBeHidden();
    });

    await test.step("the grant made no schedule for a workflow that still needs inputs", async () => {
      const after = await waitForSetup(page, expert.id);
      expect(
        after.workflows.find((w) => w.id === workflow.id)?.schedule_id,
      ).toBeNull();
      expect(
        setupSummary(await listSetupItems(page, expert.id)),
        "only the connected service's item should be gone",
      ).toEqual(setupSummary(items.filter((item) => item !== connect)));
      if (workflow.graph_id) {
        expect(await listSchedules(page, workflow.graph_id)).toEqual([]);
      }
      await backToTeam(page);
      await expect(
        setup.getByText(
          `Schedule ${inputs?.workflow_name} for ${expert.name}`,
          { exact: true },
        ),
      ).toBeVisible();
    });
  });

  test("letting an expert go archives it, and hiring it again keeps a paused schedule paused and doubles nothing", async ({
    page,
    journeyUser: _user,
  }) => {
    test.setTimeout(REHIRE_TEST_TIMEOUT);
    const roster = scheduledExpert(await listTemplates(page));
    const expert = await hireThroughAPI(page, roster.id);
    const before = await waitForSetup(page, expert.id);
    const own = await createOwnWorkflow(
      page,
      `Journey digest ${uniqueStamp()}`,
    );
    const installed = await installOnExpert(page, expert.id, own);
    const workflowIDs = [...before.workflows.map((w) => w.id), installed.id];
    const scheduleIDs: string[] = [];

    try {
      await test.step("the expert's page schedules a workflow only once its input is given", async () => {
        await page.goto(`/team/${expert.id}?tab=schedules`);
        await page
          .getByRole("button", { name: "Create schedule" })
          .first()
          .click();
        await page
          .getByRole("dialog", { name: "Create schedule" })
          .getByRole("button", { name: own.name })
          .click();
        const runDialog = page.getByRole("dialog").filter({
          has: page.getByRole("button", { name: "Schedule Task" }),
        });
        const scheduleTask = runDialog.getByRole("button", {
          name: "Schedule Task",
        });
        await expect(
          scheduleTask,
          "no schedule without the input the workflow needs",
        ).toBeDisabled();
        await runDialog
          .getByTestId(`agent-input-${own.input}`)
          .fill("This week's notes");
        await scheduleTask.click();
        const scheduleDialog = page.getByRole("dialog", {
          name: "Schedule Run",
        });
        await scheduleDialog.getByLabel("Schedule Name").fill(RUNNING_SCHEDULE);
        const created = page.waitForResponse(
          isPath("POST", `/api/proxy/api/graphs/${own.graphID}/schedules`),
        );
        await scheduleDialog
          .getByRole("button", { name: "Schedule", exact: true })
          .click();
        const response = await created;
        expect(response.ok(), "creating the schedule").toBe(true);
        scheduleIDs.push(((await response.json()) as Schedule).id);
        await expect(scheduleDialog).toBeHidden();
        scheduleIDs.push(
          (await scheduleThroughAPI(page, own, PAUSED_SCHEDULE)).id,
        );
        const schedules = await listSchedules(page, own.graphID);
        expect(
          schedules
            .map((schedule) => [schedule.name, schedule.expert_id])
            .sort(),
          "both schedules should be the expert's",
        ).toEqual([
          [RUNNING_SCHEDULE, expert.id],
          [PAUSED_SCHEDULE, expert.id],
        ]);
      });

      const pause = scriptedRun("expert-pause-schedule");
      await test.step("asked in its chat, the expert pauses one of its schedules", async () => {
        await page.goto(`/copilot?expertId=${expert.id}`);
        await sendInChat(
          page,
          page,
          `${pause.marker} Please pause the schedule called "${PAUSED_SCHEDULE}".`,
        );
        await expect(
          page.getByRole("log").getByText(PAUSE.turns[2].text ?? ""),
        ).toBeVisible({ timeout: MODEL_TURN_TIMEOUT });
        expectEachTurnOnce(await readJournal(pause.nonce), PAUSE);
        expect(await scheduleStates(page, own.graphID)).toEqual([
          [RUNNING_SCHEDULE, "active"],
          [PAUSED_SCHEDULE, "paused"],
        ]);
      });

      await test.step("letting it go archives it and pauses what was still running", async () => {
        await page.goto("/team");
        await teamCard(page, expert.name)
          .getByTestId("expert-card-actions")
          .click();
        await page
          .getByRole("menuitem", { name: `Fire ${expert.name}` })
          .click();
        const dialog = page.getByRole("dialog", {
          name: `Fire ${expert.name}?`,
        });
        await expect(
          dialog.getByText("1 automation will pause."),
        ).toBeVisible();
        await expect(dialog.getByText(RUNNING_SCHEDULE)).toBeVisible();
        const archived = page.waitForResponse(
          isPath("DELETE", `/api/proxy/api/experts/${expert.id}`),
        );
        await dialog.getByTestId("fire-expert-confirm").click();
        expect((await archived).status(), "archiving").toBe(204);
        await expect(
          page.getByRole("link", {
            name: `View ${expert.name}`,
            exact: true,
          }),
          "Team should no longer list it",
        ).toHaveCount(0);
        expect((await listHiredExperts(page)).map((e) => e.id)).not.toContain(
          expert.id,
        );
        expect(
          await listSchedules(page, own.graphID),
          "an archived expert's schedules are kept out of reach",
        ).toEqual([]);
      });

      await test.step("hiring it again from the Marketplace brings the same expert back", async () => {
        await page.goto(`/marketplace/experts/${roster.id}`);
        const hire = page.getByRole("button", {
          name: `Hire ${roster.name}`,
          exact: true,
        });
        const hired = page.waitForResponse(
          isPath("POST", "/api/proxy/api/experts"),
        );
        await hire.click();
        const response = await hired;
        expect(response.ok(), `re-hiring: ${response.status()}`).toBe(true);
        expect(
          ((await response.json()) as { expert: Expert }).expert.id,
          "the expert should come back, not a second one",
        ).toBe(expert.id);
        await page
          .getByRole("dialog")
          .getByRole("button", { name: "Skip for now" })
          .click();
        await page.waitForURL(
          (url) =>
            url.pathname === "/copilot" &&
            url.searchParams.get("expertId") === expert.id,
        );
        await expect(
          page.getByRole("log").getByText(PAUSE.turns[2].text ?? ""),
          "it should reopen the chat it had, not introduce itself again",
        ).toBeVisible();
      });

      await test.step("nothing is doubled, and the schedule paused before letting go stays paused", async () => {
        const rehired = await waitForSetup(page, expert.id);
        expect(rehired.is_archived).toBe(false);
        expect(rehired.schedules_paused_at).toBeNull();
        expect(
          rehired.workflows.map((w) => w.id).sort(),
          "each workflow should be there once",
        ).toEqual([...workflowIDs].sort());
        expect(
          (await listHiredExperts(page))
            .filter((e) => e.source_template_id === roster.id)
            .map((e) => e.id),
        ).toEqual([expert.id]);
        const states = await scheduleStates(page, own.graphID);
        expect(
          states.map(([name]) => name),
          "re-hiring should make no second schedule",
        ).toEqual([RUNNING_SCHEDULE, PAUSED_SCHEDULE]);
        expect(
          (await listSchedules(page, own.graphID))
            .map((schedule) => schedule.id)
            .sort(),
          "re-hiring should bring back the schedules it had, not new ones",
        ).toEqual([...scheduleIDs].sort());
        expect(
          stateOf(states, RUNNING_SCHEDULE),
          "re-hiring should resume what letting go paused",
        ).toBe("active");
        await deleteSchedules(page, own.graphID, scheduleIDs);
        // Guards #14577: re-hiring must not resume a schedule the user paused.
        expect(
          stateOf(states, PAUSED_SCHEDULE),
          "a schedule the user paused before letting the expert go should stay paused",
        ).toBe("paused");
      });
    } finally {
      await deleteSchedules(page, own.graphID, scheduleIDs);
    }
  });
});

/** The scripted run's chat turns: [turn, whether its reply finished]. */
async function heldTurns(nonce: string) {
  return (await readJournal(nonce))
    .filter((entry) => entry.kind === "turn")
    .map((entry) => [entry.turn, entry.completed ?? null]);
}

/** Sends a message from a chat's composer once it can take one. */
async function sendInChat(page: Page, scope: Page | Locator, message: string) {
  const input = scope.getByRole("textbox", { name: "Chat message input" });
  await expect(input).toBeEditable();
  await input.fill(message);
  const sent = page.waitForRequest(isChatStream("POST"));
  await input.press("Enter");
  await sent;
}

async function backToTeam(page: Page) {
  await page.getByTestId("expert-back-to-team").click();
  await page.waitForURL((url) => url.pathname === "/team");
}

/** From Team, an expert's Integrations tab, in-app. */
async function openIntegrations(page: Page, expertName: string) {
  await page
    .getByRole("link", { name: `View ${expertName}`, exact: true })
    .click();
  await page.waitForURL(/\/team\/[^/?]+/);
  await page.getByRole("tab", { name: "Integrations" }).click();
}

/** A service Setup needed asks to connect that takes an API key. */
async function apiKeyItem(page: Page, items: SetupItem[]) {
  const response = await page.request.get(
    "/api/proxy/api/integrations/providers",
  );
  expect(response.ok(), `listing providers: ${response.status()}`).toBe(true);
  const byKey = new Set(
    (
      (await response.json()) as {
        name: string;
        supported_auth_types: string[];
      }[]
    )
      .filter((provider) => provider.supported_auth_types.includes("api_key"))
      .map((provider) => provider.name),
  );
  const item = items
    .filter(
      (candidate) =>
        candidate.resolution === "connect" &&
        candidate.providers.length === 1 &&
        byKey.has(candidate.providers[0]),
    )
    .sort((a, b) => a.providers[0].localeCompare(b.providers[0]))[0];
  if (!item) {
    throw new Error(
      `Setup needed asks for no service a key can connect: ${JSON.stringify(items)}`,
    );
  }
  return item;
}

function stateOf(states: [string, string][], name: string) {
  return states.find(([scheduleName]) => scheduleName === name)?.[1];
}

/** Setup items as what each asks for and for which workflow, in a stable
 *  order: two workflows can ask for the same service. */
function setupSummary(items: SetupItem[]) {
  return items
    .map(
      (item) =>
        `${item.resolution}: ${item.providers.join("+")} for ${item.workflow_id}`,
    )
    .sort();
}

function escapeRegExp(text: string) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
