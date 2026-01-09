import { LoginPage } from "./pages/login.page";
import test, { expect } from "@playwright/test";
import { TEST_CREDENTIALS } from "./credentials";
import { getSelectors } from "./utils/selectors";
import { hasUrl, isHidden } from "./utils/assertion";

test.beforeEach(async ({ page }) => {
  const loginPage = new LoginPage(page);
  await page.goto("/login");
  await loginPage.login(TEST_CREDENTIALS.email, TEST_CREDENTIALS.password);
  await hasUrl(page, "/marketplace");
});

test("dashboard page loads successfully", async ({ page }) => {
  const { getText } = getSelectors(page);
  await page.goto("/profile/dashboard");

  await expect(getText("Agent dashboard")).toBeVisible();
  await expect(getText("Submit a New Agent")).toBeVisible();
  await expect(getText("Your uploaded agents")).toBeVisible();
});

test("submit agent button works correctly", async ({ page }) => {
  const { getId, getText } = getSelectors(page);

  await page.goto("/profile/dashboard");
  const submitAgentButton = getId("submit-agent-button");
  await expect(submitAgentButton).toBeVisible();
  await submitAgentButton.click();

  await expect(getText("Publish Agent")).toBeVisible();
  await expect(
    getText("Select your project that you'd like to publish"),
  ).toBeVisible();

  await page.locator('button[aria-label="Close"]').click();
  await expect(getText("Publish Agent")).not.toBeVisible();
});

test("agent table view action works correctly for rejected agents", async ({
  page,
}) => {
  await page.goto("/profile/dashboard");

  const agentTable = page.getByTestId("agent-table");
  await expect(agentTable).toBeVisible();

  const rows = agentTable.getByTestId("agent-table-row");

  // Find a row with rejected status
  const rejectedRow = rows.filter({ hasText: "Rejected" }).first();
  if (!(await rejectedRow.count())) {
    console.log("No rejected agents available; skipping view test.");
    return;
  }

  await rejectedRow.scrollIntoViewIfNeeded();

  const actionsButton = rejectedRow.getByTestId("agent-table-row-actions");
  await actionsButton.waitFor({ state: "visible", timeout: 10000 });
  await actionsButton.scrollIntoViewIfNeeded();
  await actionsButton.click();

  // View button testing
  const viewButton = page.getByRole("menuitem", { name: "View" });
  await expect(viewButton).toBeVisible();
  await viewButton.click();

  const modal = page.getByTestId("publish-agent-modal");
  await expect(modal).toBeVisible();
  const viewAgentName = modal.getByText("Agent is awaiting review");
  await expect(viewAgentName).toBeVisible();

  await page.getByRole("button", { name: "Done" }).click();
  await expect(modal).not.toBeVisible();
});

test("agent table delete action works correctly", async ({ page }) => {
  await page.goto("/profile/dashboard");

  const agentTable = page.getByTestId("agent-table");
  await expect(agentTable).toBeVisible();

  const rows = agentTable.getByTestId("agent-table-row");

  // Delete button testing — only works for PENDING submissions
  const beforeCount = await rows.count();

  if (beforeCount === 0) {
    console.log("No agents available; skipping delete flow.");
    return;
  }

  // Find a PENDING submission to delete
  const pendingRow = rows.filter({ hasText: "Pending" }).first();
  if (!(await pendingRow.count())) {
    console.log("No pending agents available; skipping delete flow.");
    return;
  }

  const deletedSubmissionId =
    await pendingRow.getAttribute("data-submission-id");
  await pendingRow.scrollIntoViewIfNeeded();

  const delActionsButton = pendingRow.getByTestId("agent-table-row-actions");
  await delActionsButton.waitFor({ state: "visible", timeout: 10000 });
  await delActionsButton.scrollIntoViewIfNeeded();
  await delActionsButton.click();

  const deleteButton = page.getByRole("menuitem", { name: "Delete" });
  await expect(deleteButton).toBeVisible();
  await deleteButton.click();

  // Assert that the card with the deleted agent ID is not visible
  await isHidden(page.locator(`[data-submission-id="${deletedSubmissionId}"]`));
});

test("edit and delete actions are unavailable for non-pending submissions", async ({
  page,
}) => {
  await page.goto("/profile/dashboard");

  const agentTable = page.getByTestId("agent-table");
  await expect(agentTable).toBeVisible();

  const rows = agentTable.getByTestId("agent-table-row");

  // Test with rejected submissions (view only)
  const rejectedRow = rows.filter({ hasText: "Rejected" }).first();
  if (await rejectedRow.count()) {
    await rejectedRow.scrollIntoViewIfNeeded();
    const actionsButton = rejectedRow.getByTestId("agent-table-row-actions");
    await actionsButton.waitFor({ state: "visible", timeout: 10000 });
    await actionsButton.scrollIntoViewIfNeeded();
    await actionsButton.click();

    await expect(page.getByRole("menuitem", { name: "View" })).toBeVisible();
    await expect(page.getByRole("menuitem", { name: "Edit" })).toHaveCount(0);
    await expect(page.getByRole("menuitem", { name: "Delete" })).toHaveCount(0);

    // Close the menu
    await page.keyboard.press("Escape");
  }

  // Test with approved submissions (view only)
  const approvedRow = rows.filter({ hasText: "Approved" }).first();
  if (await approvedRow.count()) {
    await approvedRow.scrollIntoViewIfNeeded();
    const actionsButton = approvedRow.getByTestId("agent-table-row-actions");
    await actionsButton.waitFor({ state: "visible", timeout: 10000 });
    await actionsButton.scrollIntoViewIfNeeded();
    await actionsButton.click();

    await expect(page.getByRole("menuitem", { name: "View" })).toBeVisible();
    await expect(page.getByRole("menuitem", { name: "Edit" })).toHaveCount(0);
    await expect(page.getByRole("menuitem", { name: "Delete" })).toHaveCount(0);
  }
});

test("editing a pending submission works correctly", async ({ page }) => {
  await page.goto("/profile/dashboard");

  const agentTable = page.getByTestId("agent-table");
  await expect(agentTable).toBeVisible();

  const rows = agentTable.getByTestId("agent-table-row");

  // Find a PENDING submission to edit (only PENDING submissions can be edited)
  const pendingRow = rows.filter({ hasText: "Pending" }).first();
  if (!(await pendingRow.count())) {
    console.log("No pending agents available; skipping edit test.");
    return;
  }

  const beforeCount = await rows.count();

  await pendingRow.scrollIntoViewIfNeeded();
  const actionsButton = pendingRow.getByTestId("agent-table-row-actions");
  await actionsButton.waitFor({ state: "visible", timeout: 10000 });
  await actionsButton.scrollIntoViewIfNeeded();
  await actionsButton.click();

  const editButton = page.getByRole("menuitem", { name: "Edit" });
  await expect(editButton).toBeVisible();
  await editButton.click();

  const editModal = page.getByTestId("edit-agent-modal");
  await expect(editModal).toBeVisible();

  const newTitle = `E2E Edit Pending ${Date.now()}`;
  await page.getByRole("textbox", { name: "Title" }).fill(newTitle);
  await page
    .getByRole("textbox", { name: "Changes Summary" })
    .fill("E2E change - updating pending submission");

  await page.getByRole("button", { name: "Update submission" }).click();
  await expect(editModal).not.toBeVisible();

  // A new submission should appear with pending state
  await expect(async () => {
    const afterCount = await rows.count();
    expect(afterCount).toBeGreaterThan(beforeCount);
  }).toPass();

  const newRow = rows.filter({ hasText: newTitle }).first();
  await expect(newRow).toBeVisible();
  await expect(newRow).toContainText(/Awaiting review/);
});

test("editing a pending agent updates the same submission in place", async ({
  page,
}) => {
  await page.goto("/profile/dashboard");

  const agentTable = page.getByTestId("agent-table");
  await expect(agentTable).toBeVisible();

  const rows = agentTable.getByTestId("agent-table-row");

  const pendingRow = rows.filter({ hasText: /Awaiting review/ }).first();
  if (!(await pendingRow.count())) {
    console.log("No pending agents available; skipping pending edit test.");
    return;
  }

  const beforeCount = await rows.count();

  await pendingRow.scrollIntoViewIfNeeded();
  const actionsButton = pendingRow.getByTestId("agent-table-row-actions");
  await actionsButton.waitFor({ state: "visible", timeout: 10000 });
  await actionsButton.scrollIntoViewIfNeeded();
  await actionsButton.click();

  const editButton = page.getByRole("menuitem", { name: "Edit" });
  await expect(editButton).toBeVisible();
  await editButton.click();

  const editModal = page.getByTestId("edit-agent-modal");
  await expect(editModal).toBeVisible();

  const newTitle = `E2E Edit Pending ${Date.now()}`;
  await page.getByRole("textbox", { name: "Title" }).fill(newTitle);
  await page
    .getByRole("textbox", { name: "Changes Summary" })
    .fill("E2E change - pending -> same submission");

  await page.getByRole("button", { name: "Update submission" }).click();
  await expect(editModal).not.toBeVisible();

  // Count should remain the same
  await expect(async () => {
    const afterCount = await rows.count();
    expect(afterCount).toBe(beforeCount);
  }).toPass();

  const updatedRow = rows.filter({ hasText: newTitle }).first();
  await expect(updatedRow).toBeVisible();
  await expect(updatedRow).toContainText(/Awaiting review/);
});
