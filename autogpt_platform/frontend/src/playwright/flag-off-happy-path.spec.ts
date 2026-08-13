import { expect, test } from "./coverage-fixture";
import { E2E_AUTH_STATES } from "./credentials/accounts";
import { CopilotPage } from "./pages/copilot.page";

test.use({ storageState: E2E_AUTH_STATES.marketplace });

test("flag-off: marketplace hides the AI experts section", async ({ page }) => {
  test.setTimeout(30000);

  await page.goto("/marketplace");
  await page.waitForLoadState("domcontentloaded");

  await expect(
    page.getByText(
      "Bringing you AI agents designed by thinkers from around the world",
    ),
    "flag-off hero must render when HIRE_EXPERTS is off",
  ).toBeVisible({ timeout: 15000 });

  await expect(
    page.getByText("Meet the AI Experts"),
    "experts section must be absent when HIRE_EXPERTS is off",
  ).toHaveCount(0);
});

test("flag-off: copilot hides the workspace panel", async ({ page }) => {
  test.setTimeout(30000);

  const copilotPage = new CopilotPage(page);
  const sessionId = await copilotPage.createSessionViaApi();
  await copilotPage.open(sessionId);
  await copilotPage.waitForChatInput();

  await expect(
    page.getByLabel("Open workspace panel"),
    "workspace panel toggle must be absent when ARTIFACTS is off",
  ).toHaveCount(0);
});

test("flag-off: library shows the agent briefing recap", async ({ page }) => {
  test.setTimeout(30000);

  await page.goto("/library");
  await page.waitForLoadState("domcontentloaded");

  await expect(
    page.getByTestId("library-textbox").first(),
    "library page must render under flag-off",
  ).toBeVisible({ timeout: 15000 });

  await expect(
    page.getByText("Agent Briefing"),
    "agent briefing recap must render when AGENT_BRIEFING is on (its flag-off default)",
  ).toBeVisible({ timeout: 15000 });
});

// This verifies the hard-coded defaultFlags map, not real LaunchDarkly-off
// behaviour: under NEXT_PUBLIC_PW_TEST=true useFlagStatus short-circuits to
// defaultFlags[flag]. The unit suite (team/__tests__/main.test.tsx) and the
// release checklist's LD flag audit cover the production off-state.
test("flag-off: /team returns 404 when HIRE_EXPERTS is off", async ({
  page,
}) => {
  test.setTimeout(15000);

  const response = await page.goto("/team");

  expect(response?.status()).toBe(404);

  await expect(page).toHaveURL(/\/team/);

  await expect(
    page.getByText(/could not be found|not found/i).first(),
    "Next.js 404 page must render when HIRE_EXPERTS is off",
  ).toBeVisible({ timeout: 10000 });
});
