import { expect, test } from "./coverage-fixture";
import { E2E_AUTH_STATES } from "./credentials/accounts";
import { CopilotPage } from "./pages/copilot.page";

test.use({ storageState: E2E_AUTH_STATES.marketplace });

test("copilot happy path: user can create a deterministic AutoPilot session and keep it after reload", async ({
  page,
}) => {
  test.setTimeout(120000);

  const copilotPage = new CopilotPage(page);
  await copilotPage.open();

  const sessionId = await copilotPage.createSessionViaApi();

  await copilotPage.open(sessionId);
  await copilotPage.waitForChatInput();

  await page.reload();
  await page.waitForLoadState("domcontentloaded");
  await copilotPage.dismissNotificationPrompt();

  await expect
    .poll(() => new URL(page.url()).searchParams.get("sessionId"), {
      timeout: 15000,
    })
    .toBe(sessionId);
  await copilotPage.waitForChatInput();

  // Sending a message must render the user's prompt in the conversation
  // immediately. This catches a regression where the chat input accepts
  // text but Enter is a no-op, without depending on knowing the exact
  // backend endpoint name (which has shifted historically).
  const userPrompt = `ping from e2e ${Date.now().toString().slice(-6)}`;
  const chatInput = copilotPage.getChatInput();
  await chatInput.fill(userPrompt);
  await chatInput.press("Enter");

  await expect(
    page.getByText(userPrompt, { exact: false }).first(),
    "user's typed prompt must appear in the chat after pressing Enter",
  ).toBeVisible({ timeout: 15000 });
});
