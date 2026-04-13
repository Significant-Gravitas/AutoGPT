import type { Page } from "@playwright/test";
import { expect, test } from "./coverage-fixture";
import { E2E_AUTH_STATES } from "./credentials/accounts";
import { CopilotPage } from "./pages/copilot.page";

test.use({ storageState: E2E_AUTH_STATES.parallelA });

function getSessionIdFromUrl(page: Page) {
  return new URL(page.url()).searchParams.get("sessionId");
}

async function waitForStreamRequest(page: Page, expectedMessage: string) {
  const request = await page.waitForRequest(
    (candidate) =>
      candidate.method() === "POST" &&
      candidate.url().includes("/api/chat/sessions/") &&
      candidate.url().endsWith("/stream"),
    { timeout: 20000 },
  );

  const payload = request.postDataJSON() as {
    message?: string;
    is_user_message?: boolean;
  };

  expect(payload.message).toBe(expectedMessage);
  expect(payload.is_user_message).toBe(true);
}

async function waitForStartedConversation(page: Page, prompt: string) {
  await expect
    .poll(() => getSessionIdFromUrl(page), { timeout: 15000 })
    .not.toBeNull();
  await expect(page.getByText(prompt, { exact: true }).first()).toBeVisible({
    timeout: 15000,
  });
}

async function deleteSession(page: Page) {
  const sessionId = getSessionIdFromUrl(page);
  if (!sessionId) {
    return;
  }

  const response = await page.request.delete(
    `/api/proxy/api/chat/sessions/${sessionId}`,
  );
  expect(response.ok()).toBe(true);
}

test("copilot happy path: user can start a new AutoPilot conversation via prompt", async ({
  page,
}) => {
  test.setTimeout(90000);

  const copilotPage = new CopilotPage(page);
  const prompt = `What should I automate first? ${Date.now()}`;

  await copilotPage.open();
  await copilotPage.waitForEmptyChatInput();

  const streamRequestPromise = waitForStreamRequest(page, prompt);
  await copilotPage.submitEmptyChatPrompt(prompt);

  await streamRequestPromise;
  await waitForStartedConversation(page, prompt);
  await deleteSession(page);
});

test("copilot happy path: user can start a new AutoPilot conversation via suggestion", async ({
  page,
}) => {
  test.setTimeout(90000);

  const copilotPage = new CopilotPage(page);
  const themeName = "Learn";
  const prompt = "What can AutoGPT do for me?";

  await copilotPage.open();
  await copilotPage.waitForEmptyChatInput();

  await page.getByRole("button", { name: themeName, exact: true }).click();

  const streamRequestPromise = waitForStreamRequest(page, prompt);
  await page.getByRole("button", { name: prompt, exact: true }).click();

  await streamRequestPromise;
  await waitForStartedConversation(page, prompt);
  await deleteSession(page);
});
