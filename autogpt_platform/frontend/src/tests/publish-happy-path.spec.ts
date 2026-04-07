import { expect, test } from "./coverage-fixture";
import { getTestUserWithLibraryAgents } from "./credentials";
import { LoginPage } from "./pages/login.page";

test("publish happy path: user can submit an agent for marketplace review", async ({
  page,
}) => {
  test.setTimeout(120000);

  const loginPage = new LoginPage(page);
  const richUser = getTestUserWithLibraryAgents();

  await page.goto("/login");
  await loginPage.login(richUser.email, richUser.password);
  await expect(page).toHaveURL(/\/marketplace/);

  await page.goto("/marketplace");
  await page.getByRole("button", { name: "Become a Creator" }).click();

  const publishAgentModal = page.getByTestId("publish-agent-modal");
  await expect(publishAgentModal).toBeVisible();
  await expect(
    publishAgentModal.getByText(
      "Select your project that you'd like to publish",
    ),
  ).toBeVisible();

  await publishAgentModal.getByTestId("agent-card").first().click();
  await publishAgentModal
    .getByRole("button", { name: "Next", exact: true })
    .click();

  await expect(
    publishAgentModal.getByText("Write a bit of details about your agent"),
  ).toBeVisible();

  const suffix = Date.now().toString().slice(-6);

  await publishAgentModal.getByLabel("Title").fill(`Publish Flow ${suffix}`);
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
});
