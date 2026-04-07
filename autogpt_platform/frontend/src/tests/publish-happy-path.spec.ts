import { Page } from "@playwright/test";
import { expect, test } from "./coverage-fixture";
import { getSeededTestUser } from "./credentials/accounts";
import { BuildPage } from "./pages/build.page";
import { LibraryPage } from "./pages/library.page";
import { LoginPage } from "./pages/login.page";

function createUniqueAgentName(prefix: string) {
  return `${prefix} ${Date.now().toString().slice(-6)}`;
}

async function logInPublishUser(page: Page) {
  const loginPage = new LoginPage(page);
  const publishUser = getSeededTestUser("parallelA");

  await page.goto("/login");
  await loginPage.login(publishUser.email, publishUser.password);
  await expect(page).toHaveURL(/\/marketplace/);
}

async function addSimpleAgentBlocks(buildPage: BuildPage) {
  await buildPage.addBlockByClick("Store Value");
  await buildPage.waitForNodeOnCanvas(1);
  await buildPage.fillBlockInputByPlaceholder(
    "Enter string value...",
    "publish-value",
    0,
  );

  await buildPage.addBlockByClick("Add to Dictionary");
  await buildPage.waitForNodeOnCanvas(2);

  const dictionaryInputs = buildPage
    .getNodeLocator(1)
    .locator('input[placeholder="Enter string value..."]');
  await dictionaryInputs.nth(0).fill("publish-key");
  await dictionaryInputs.nth(1).fill("publish-value");
}

async function createPublishableAgent(page: Page) {
  const buildPage = new BuildPage(page);
  const libraryPage = new LibraryPage(page);
  const agentName = createUniqueAgentName("Publish Flow Agent");

  await page.goto("/build");
  await page.waitForLoadState("domcontentloaded");
  await buildPage.closeTutorial();
  await expect(page.locator(".react-flow")).toBeVisible({ timeout: 15000 });
  await expect(page.getByTestId("blocks-control-blocks-button")).toBeVisible({
    timeout: 15000,
  });

  await addSimpleAgentBlocks(buildPage);
  await buildPage.saveAgent(agentName, "PR E2E publish coverage");
  await buildPage.waitForSaveComplete();
  await buildPage.waitForSaveButton();

  await page.goto("/library");
  await libraryPage.waitForAgentsToLoad();
  await libraryPage.searchAgents(agentName);
  await libraryPage.waitForAgentsToLoad();

  const createdAgent = page
    .getByTestId("library-agent-card")
    .filter({ hasText: agentName })
    .first();
  await expect(createdAgent).toBeVisible({ timeout: 15000 });

  return agentName;
}

async function submitAgentForReview(page: Page) {
  await logInPublishUser(page);
  const publishableAgentName = await createPublishableAgent(page);

  await page.goto("/marketplace");
  await page.getByRole("button", { name: "Become a Creator" }).click();

  const publishAgentModal = page.getByTestId("publish-agent-modal");
  await expect(publishAgentModal).toBeVisible();
  await expect(
    publishAgentModal.getByText(
      "Select your project that you'd like to publish",
    ),
  ).toBeVisible();

  const publishableAgentCard = publishAgentModal
    .getByTestId("agent-card")
    .filter({ hasText: publishableAgentName })
    .first();
  await expect(publishableAgentCard).toBeVisible({ timeout: 15000 });
  await publishableAgentCard.click();
  await publishAgentModal
    .getByRole("button", { name: "Next", exact: true })
    .click();

  await expect(
    publishAgentModal.getByText("Write a bit of details about your agent"),
  ).toBeVisible();

  const suffix = Date.now().toString().slice(-6);
  const agentTitle = `Publish Flow ${suffix}`;

  await publishAgentModal.getByLabel("Title").fill(agentTitle);
  await publishAgentModal
    .getByLabel("Subheader")
    .fill("A deterministic marketplace submission");
  await publishAgentModal.getByLabel("Slug").fill(`publish-flow-${suffix}`);
  await publishAgentModal
    .getByLabel("YouTube video link")
    .fill("https://www.youtube.com/watch?v=test123");

  await publishAgentModal.getByRole("combobox", { name: "Category" }).click();
  await page.getByRole("option", { name: "Other" }).click();

  await publishAgentModal
    .getByLabel("Description")
    .fill("A deterministic publish flow for consolidated Playwright coverage.");

  const submitButton = publishAgentModal.getByRole("button", {
    name: "Submit for review",
  });
  await expect(submitButton).toBeEnabled();
  await submitButton.click();

  await expect(
    publishAgentModal.getByText("Agent is awaiting review"),
  ).toBeVisible();
  await expect(
    publishAgentModal.getByTestId("view-progress-button"),
  ).toBeVisible();

  return agentTitle;
}

async function waitForDashboardSubmission(page: Page, agentTitle: string) {
  for (let attempt = 0; attempt < 3; attempt += 1) {
    const submissionRow = page
      .getByTestId("agent-table-row")
      .filter({ hasText: agentTitle })
      .first();

    if (await submissionRow.isVisible().catch(() => false)) {
      return submissionRow;
    }

    await page.reload();
    await expect(page).toHaveURL(/\/profile\/dashboard/);
    await expect(page.getByText("Agent dashboard")).toBeVisible();
  }

  throw new Error(`Submission row for "${agentTitle}" did not appear`);
}

test("publish happy path: user can submit an agent for marketplace review and track it from the dashboard", async ({
  page,
}) => {
  test.setTimeout(120000);

  const agentTitle = await submitAgentForReview(page);

  await page.getByTestId("view-progress-button").click();
  await expect(page).toHaveURL(/\/profile\/dashboard/);
  await expect(page.getByText("Agent dashboard")).toBeVisible();

  const submissionRow = await waitForDashboardSubmission(page, agentTitle);
  await submissionRow.getByTestId("agent-table-row-actions").click();
  await expect(page.getByRole("menuitem", { name: "Edit" })).toBeVisible();
});
