import type { BrowserContext, Locator, Page } from "@playwright/test";
import { expect, test } from "../coverage-fixture";
import { BuildPage, createUniqueAgentName } from "../pages/build.page";
import {
  clickRunButton,
  dismissFeedbackDialog,
  LibraryPage,
} from "../pages/library.page";

// A self-hoster's first hour on a fresh install with no model keys and no
// hosted services: sign up, onboard, build and run an agent, look around
// Settings. Nothing hosted-only may show up on the way, and the browser may
// talk to no host but the instance itself.
//
// Run it against a self-hosted build (BEHAVE_AS=LOCAL, payments off, no LLM
// key), e.g. the single-container appliance:
//   PLAYWRIGHT_BASE_URL=http://localhost:3400 \
//   PLAYWRIGHT_SELFHOST_EMAIL_DOMAIN=patrol.example.com \
//     pnpm exec playwright test --config playwright.journeys.config.ts selfhost
// The email domain must be on the instance's AUTH_SIGNUP_ALLOWLIST.

const EMAIL_DOMAIN =
  process.env.PLAYWRIGHT_SELFHOST_EMAIL_DOMAIN ?? "patrol.example.com";
const PASSWORD = `Selfhost-${Date.now()}-journey`;

test.describe.configure({ mode: "serial" });
test.setTimeout(300_000);

test("selfhost journey: a self-hoster's first hour has nothing hosted-only in it", async ({
  page,
  context,
}) => {
  const thirdPartyRequests = recordThirdPartyRequests(context);
  const runID = Date.now();
  // New users are asked about browser notifications on their first visit.
  await page.addLocatorHandler(
    page.getByRole("dialog", { name: "Stay in the loop" }),
    async () => {
      await page.getByRole("button", { name: "Not now" }).click();
    },
  );

  await test.step("an email off the allowlist gets the operator's answer, not the hosted waitlist", async () => {
    await submitSignup(page, `stranger-${runID}@not-allowed.example.org`);
    await expect
      .soft(page.getByText(/only allows approved emails/i).first())
      .toBeVisible();
    await expect
      .soft(page.getByText(/closed beta|join waitlist/i))
      .toHaveCount(0);
  });

  await test.step("sign up and finish onboarding with no paywall", async () => {
    await submitSignup(page, `selfhost-${runID}@${EMAIL_DOMAIN}`);
    await page.waitForURL((url) => !url.pathname.startsWith("/signup"));
    await finishOnboarding(page);
    await expect(page).toHaveURL(/\/(home|copilot)/);
  });

  await test.step("home shows no credits, billing or developer tooling", async () => {
    await expectNoHostedOnlyChrome(page);
  });

  await test.step("chat with no model configured says how to fix it, and keeps saying it", async () => {
    const notice = page.getByRole("status", { name: "No AI connection" });
    await expect.soft(notice).toBeVisible();
    await expect
      .soft(notice.getByRole("link", { name: /setup guide/i }))
      .toHaveAttribute("href", /single-container\.md#models-and-memory/);

    const composer = page.getByRole("textbox").first();
    await composer.fill("Hi there");
    await composer.press("Enter");
    const toast = page
      .getByText(/No AI connection is set up|Couldn't send message/)
      .first();
    await expect(toast).toBeVisible();
    await expect
      .soft(page.getByText(/chat_transport_not_configured/))
      .toHaveCount(0);
    await expect(composer).toHaveValue("Hi there");

    await expect(toast).toBeHidden({ timeout: 20_000 });
    await expect.soft(notice).toBeVisible();
  });

  await test.step("builder chat with no model says how to fix it, and saves no agent", async () => {
    const buildPage = new BuildPage(page);
    await buildPage.open();
    await page.getByRole("button", { name: "Chat with builder" }).click();
    const panel = page.getByRole("complementary", {
      name: "Builder chat panel",
    });
    const notice = panel.getByRole("status", { name: "No AI connection" });
    await expect.soft(notice).toBeVisible();
    await expect
      .soft(notice.getByRole("link", { name: /setup guide/i }))
      .toHaveAttribute("href", /single-container\.md#models-and-memory/);
    await expect
      .soft(panel.getByRole("button", { name: "Retry" }))
      .toHaveCount(0);

    await page.getByRole("button", { name: "Close chat" }).click();
    expect
      .soft(
        await libraryAgentNames(page),
        "opening the builder chat must not save an empty agent",
      )
      .toEqual([]);
  });

  const agentName = createUniqueAgentName("Selfhost Journey Agent");
  const expectedOutput = `selfhost-output-${runID}`;
  const outputName = `selfhost-result-${runID}`;

  await test.step("build a tiny agent in the builder", async () => {
    const buildPage = new BuildPage(page);
    await buildPage.open();
    await buildPage.addBlockByClick("Store Value");
    await buildPage.waitForNodeOnCanvas(1);
    await buildPage.fillBlockInputByPlaceholder(
      "Enter string value...",
      expectedOutput,
      0,
    );
    await buildPage.addBlockByClick("Agent Output");
    await buildPage.waitForNodeOnCanvas(2);
    await connectHandles(page, "output", "value");
    await expect.poll(() => buildPage.getEdgeCount()).toBe(1);
    await buildPage.fillLastNodeTextInput("Agent Output", outputName);
    await expect.soft(page.locator(AGENTATION)).toHaveCount(0);
    await buildPage.saveAgent(agentName, "Self-hosted first-hour journey");
    await buildPage.waitForSaveComplete();
    await buildPage.waitForSaveButton();
  });

  await test.step("run it from the Library and read its output, with no cost shown", async () => {
    const libraryPage = new LibraryPage(page);
    await libraryPage.openSavedAgent(agentName);
    await clickRunButton(page);
    await libraryPage.waitForRunToComplete();
    await dismissFeedbackDialog(page);
    await libraryPage.assertRunOutputValue(outputName, expectedOutput);
    await expect.soft(page.getByText(/^Cost: \$/)).toHaveCount(0);
    await expectNoHostedOnlyChrome(page);
  });

  await test.step("the Library's briefing shows no spend", async () => {
    await page.goto("/library");
    await expect(page.getByText("Workflow Briefing")).toBeVisible();
    await expect.soft(page.getByText("Spent this month")).toHaveCount(0);
    await expect
      .soft(page.getByRole("button", { name: /costs breakdown/i }))
      .toHaveCount(0);
  });

  await test.step("settings: memory is reachable, billing is not", async () => {
    await page.goto("/settings/memory");
    await expect
      .soft(page.getByRole("heading", { name: "Memory", level: 1 }))
      .toBeVisible();

    await page.goto("/settings/account");
    await expect(page.getByRole("link", { name: "Account" })).toBeVisible();
    await expect
      .soft(page.getByRole("link", { name: "Billing" }))
      .toHaveCount(0);

    await page.goto("/settings/billing");
    await expect
      .soft(page.getByText("This page could not be found.").first())
      .toBeVisible();
    await expect
      .soft(page.getByText(/pick a plan|get pro|add credits/i))
      .toHaveCount(0);
  });

  expect(
    [...thirdPartyRequests].sort(),
    "a self-hosted install must not make the browser call other hosts",
  ).toEqual([]);
});

const AGENTATION = "[data-agentation-root], [data-agentation-toolbar]";

// Drags from the first node's named output to the second node's named input.
// BuildPage.connectNodes takes each node's first handle, which for Store Value
// is ``error``.
async function connectHandles(page: Page, output: string, input: string) {
  const nodeIDs = await page
    .locator('[data-id^="custom-node-"]')
    .evaluateAll((nodes) =>
      nodes.map((node) =>
        (node.getAttribute("data-id") ?? "").replace("custom-node-", ""),
      ),
    );
  const source = page.locator(
    `[data-tutorial-id="output-handler-${nodeIDs[0]}-${output}"]`,
  );
  const target = page.locator(
    `[data-tutorial-id="input-handler-${nodeIDs[1]}-${input}"]`,
  );
  const from = await centerOf(source);
  const to = await centerOf(target);
  await page.mouse.move(from.x, from.y);
  await page.mouse.down();
  const steps = 20;
  for (let step = 1; step <= steps; step++) {
    await page.mouse.move(
      from.x + ((to.x - from.x) * step) / steps,
      from.y + ((to.y - from.y) * step) / steps,
    );
  }
  await page.mouse.up();
}

async function libraryAgentNames(page: Page) {
  const response = await page.request.get(
    "/api/proxy/api/library/agents?page_size=100",
  );
  expect(response.ok()).toBe(true);
  const { agents } = (await response.json()) as { agents: { name: string }[] };
  return agents.map((agent) => agent.name);
}

async function centerOf(locator: Locator) {
  const box = await locator.boundingBox();
  if (!box) throw new Error("handle is not on screen");
  return { x: box.x + box.width / 2, y: box.y + box.height / 2 };
}

async function submitSignup(page: Page, email: string) {
  await page.goto("/signup");
  await expect(page.getByText("Create your account")).toBeVisible();
  await page.getByLabel("Email").fill(email);
  await page.locator("#password").fill(PASSWORD);
  await page.locator("#confirmPassword").fill(PASSWORD);
  await page.getByRole("button", { name: "Sign up" }).click();
}

// Role, pain points and the self-hosted "connect" step, then the workspace
// is prepared and the app lands on Home. A plan or paywall step fails here.
async function finishOnboarding(page: Page) {
  await expect(
    page.getByRole("heading", { name: "What best describes you?" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Engineering", exact: true }).click();
  await page.getByRole("button", { name: "Next", exact: true }).click();

  await expect(
    page.getByRole("heading", { name: /What's eating your time\?/ }),
  ).toBeVisible();
  await page.getByText("Research", { exact: true }).click();
  await page.getByRole("button", { name: "Continue", exact: true }).click();

  await expect(
    page.getByRole("heading", {
      name: "Already paying for an AI subscription?",
    }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Next", exact: true }).click();

  await page.waitForURL((url) => !url.pathname.startsWith("/onboarding"), {
    timeout: 60_000,
  });
}

async function expectNoHostedOnlyChrome(page: Page) {
  const accountMenu = page.getByTestId("profile-popout-menu-trigger");
  await expect(accountMenu).toBeVisible();
  await expect.soft(page.getByText(/earn credits/i)).toHaveCount(0);
  await expect.soft(page.locator(AGENTATION)).toHaveCount(0);

  await accountMenu.click();
  const menu = page.getByTestId("account-menu-popover");
  await expect(menu.getByText("Settings")).toBeVisible();
  await expect.soft(menu.getByText("Billing")).toHaveCount(0);
  await page.keyboard.press("Escape");
}

function recordThirdPartyRequests(context: BrowserContext) {
  const ownOrigin = new URL(
    process.env.PLAYWRIGHT_BASE_URL ?? "http://localhost:3000",
  ).origin;
  const seen = new Set<string>();
  context.on("request", (request) => {
    const url = new URL(request.url());
    if (!["http:", "https:", "ws:", "wss:"].includes(url.protocol)) return;
    const origin = url.origin.replace(/^ws/, "http");
    if (origin === ownOrigin) return;
    seen.add(`${request.method()} ${url.host}${url.pathname}`);
  });
  return seen;
}
