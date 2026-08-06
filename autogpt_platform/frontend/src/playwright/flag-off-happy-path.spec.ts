import { expect, test } from "./coverage-fixture";
import { E2E_AUTH_STATES } from "./credentials/accounts";
import { CopilotPage } from "./pages/copilot.page";

test.use({ storageState: E2E_AUTH_STATES.marketplace });

test("flag-off: copilot loads and accepts input under flag-off", async ({
  page,
}) => {
  test.setTimeout(60000);

  const copilotPage = new CopilotPage(page);
  await copilotPage.open();
  await copilotPage.waitForChatInput();

  const userPrompt = `smoke ${Date.now().toString().slice(-6)}`;
  const chatInput = copilotPage.getChatInput();
  await chatInput.fill(userPrompt);
  await chatInput.press("Enter");

  await expect(
    page.getByText(userPrompt, { exact: false }).first(),
    "user's typed prompt must appear after pressing Enter under flag-off",
  ).toBeVisible({ timeout: 15000 });
});

test("flag-off: marketplace loads under flag-off", async ({ page }) => {
  test.setTimeout(30000);

  await page.goto("/marketplace");
  await page.waitForLoadState("domcontentloaded");

  await expect(
    page.getByRole("heading", { name: "Explore AI agents" }).first(),
    "marketplace landing page must render under flag-off",
  ).toBeVisible({ timeout: 10000 });
});

test("flag-off: library loads under flag-off", async ({ page }) => {
  test.setTimeout(30000);

  await page.goto("/library");
  await page.waitForLoadState("domcontentloaded");

  await expect(
    page.getByTestId("library-textbox").first(),
    "library page must render under flag-off",
  ).toBeVisible({ timeout: 10000 });
});

test("flag-off: /team returns 404 when HIRE_EXPERTS is off", async ({
  page,
}) => {
  test.setTimeout(15000);

  await page.goto("/team");

  // notFound() keeps the URL at /team, renders the default 404 page.
  await expect(page).toHaveURL(/\/team/);

  await expect(
    page.getByText(/could not be found|not found/i).first(),
    "Next.js 404 page must render when HIRE_EXPERTS is off",
  ).toBeVisible({ timeout: 10000 });
});
