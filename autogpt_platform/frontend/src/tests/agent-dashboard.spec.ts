import { LoginPage } from "./pages/login.page";
import test, { expect } from "@playwright/test";
import { TEST_AGENT_DATA, TEST_CREDENTIALS } from "./credentials";
import { getSelectors } from "./utils/selectors";
import { hasUrl } from "./utils/assertion";

test.describe("Agent Dashboard", () => {
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

  test("agent table displays data correctly", async ({ page }) => {
    const { getText } = getSelectors(page);
    await page.goto("/profile/dashboard");

    await expect(getText("Agent info")).toBeVisible();
    await expect(getText("Date submitted")).toBeVisible();

    await expect(getText(TEST_AGENT_DATA.name).first()).toBeVisible();
    await expect(getText(TEST_AGENT_DATA.description).first()).toBeVisible();
  });

  test("agent table actions work correctly", async ({ page }) => {
    await page.goto("/profile/dashboard");
  
    const agentTable = page.getByTestId("agent-table");
    await expect(agentTable).toBeVisible();
  
    const rows = agentTable.getByTestId("agent-table-row");
  
    const testRow = rows.filter({ hasText: TEST_AGENT_DATA.name }).first();
    await testRow.scrollIntoViewIfNeeded();
  
    const actionsButton = testRow.getByTestId("agent-table-row-actions");
    await actionsButton.waitFor({ state: "visible", timeout: 10000 });
    await actionsButton.scrollIntoViewIfNeeded();
    await actionsButton.click();
  
    // View button testing
    const viewButton = page.getByRole("menuitem", { name: "View" });
    await expect(viewButton).toBeVisible();
    await viewButton.click();
  
    const modal = page.getByTestId("publish-agent-modal");
    await expect(modal).toBeVisible();
    const viewAgentName = page.getByTestId("view-agent-name");
    await expect(viewAgentName).toBeVisible();
    await expect(viewAgentName).toHaveText(TEST_AGENT_DATA.name);
  
    await page.getByRole("button", { name: "Done" }).click();
    await expect(modal).not.toBeVisible();
  
    // Delete button testing
    const beforeCount = await rows.count();
  
    let deletableIndex = -1;
    for (let i = 0; i < beforeCount; i++) {
      const nameAttr = await rows.nth(i).getAttribute("data-agent-name");
      if (nameAttr && nameAttr !== TEST_AGENT_DATA.name) {
        deletableIndex = i;
        break;
      }
    }
  
    if (deletableIndex === -1) {
      console.log("Only the test agent exists; skipping delete flow.");
      return;
    }
  
    const deletableRow = rows.nth(deletableIndex);
    await deletableRow.scrollIntoViewIfNeeded();
  
    const delActionsButton = deletableRow.getByTestId("agent-table-row-actions");
    await delActionsButton.waitFor({ state: "visible", timeout: 10000 });
    await delActionsButton.scrollIntoViewIfNeeded();
    await delActionsButton.click();
  
    const deleteButton = page.getByRole("menuitem", { name: "Delete" });
    await expect(deleteButton).toBeVisible();
    await deleteButton.click();
  
    await expect.poll(async () => await rows.count(), { timeout: 15000 }).toBe(
      beforeCount - 1,
    );
  });
});
