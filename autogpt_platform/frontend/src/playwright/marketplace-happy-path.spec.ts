import type { Page } from "@playwright/test";
import { expect, test } from "./coverage-fixture";
import { E2E_AUTH_STATES } from "./credentials/accounts";
import { waitForAgentPageLoad } from "./pages/library.page";
import { MarketplacePage } from "./pages/marketplace.page";

test.use({ storageState: E2E_AUTH_STATES.parallelB });

const DETERMINISTIC_AGENT_NAME = "E2E Calculator Agent";
const DETERMINISTIC_CREATOR_NAME = "E2E Marketplace Creator";

function getLibraryAgentIdFromUrl(url: string) {
  const match = url.match(/\/library\/agents\/([^/?#]+)/);
  return match?.[1] ?? null;
}

async function ensureDeterministicAgentIsNotInLibrary(page: Page) {
  const addLibraryButton = page.getByTestId("agent-add-library-button");
  const buttonLabel = (await addLibraryButton.textContent())?.trim() ?? "";

  if (!/See runs/i.test(buttonLabel)) {
    return;
  }

  await addLibraryButton.click();
  await expect(page).toHaveURL(/\/library\/agents\//);

  const existingLibraryAgentId = getLibraryAgentIdFromUrl(page.url());
  expect(existingLibraryAgentId).toBeTruthy();

  const deleteResponse = await page.request.delete(
    `/api/proxy/api/library/agents/${existingLibraryAgentId}`,
  );
  expect(deleteResponse.ok()).toBe(true);

  const marketplacePage = new MarketplacePage(page);
  await marketplacePage.openRunnableAgent();
  await expect(addLibraryButton).toHaveText(/Add to library/i);
}

test("marketplace happy path: user can search Marketplace and navigate to agent and creator pages", async ({
  page,
}) => {
  test.setTimeout(90000);

  const marketplacePage = new MarketplacePage(page);
  await marketplacePage.goto(page);
  await marketplacePage.searchFor(DETERMINISTIC_AGENT_NAME, page);
  await marketplacePage.waitForSearchResults();

  const agentCard = page.locator(
    `[role="button"][aria-label="${DETERMINISTIC_AGENT_NAME} agent card"]:visible`,
  );
  await expect(agentCard).toBeVisible({ timeout: 15000 });
  await agentCard.first().click();

  await expect(page).toHaveURL(
    /\/marketplace\/agent\/e2e-marketplace\/e2e-calculator-agent$/,
  );
  await expect(page.getByTestId("agent-title")).toContainText(
    DETERMINISTIC_AGENT_NAME,
  );

  const creatorLink = page.locator(
    '[data-testid="agent-creator"] a[href="/marketplace/creator/e2e-marketplace"]',
  );
  await expect(creatorLink).toBeVisible({ timeout: 15000 });
  await creatorLink.click();

  await expect(page).toHaveURL(/\/marketplace\/creator\/e2e-marketplace$/);
  await expect(page.getByTestId("creator-title")).toContainText(
    DETERMINISTIC_CREATOR_NAME,
  );
  await expect(page.getByTestId("creator-description")).toBeVisible();
});

test("marketplace happy path: user can add a Marketplace agent to Library", async ({
  page,
}) => {
  test.setTimeout(120000);

  const marketplacePage = new MarketplacePage(page);
  await marketplacePage.openRunnableAgent();
  await ensureDeterministicAgentIsNotInLibrary(page);

  let libraryAgentId: string | null = null;

  try {
    await expect(page.getByTestId("agent-add-library-button")).toHaveText(
      /Add to library/i,
    );
    await page.getByTestId("agent-add-library-button").click();
    await expect(
      page.getByText("Redirecting to your library..."),
    ).toBeVisible();
    await expect(page).toHaveURL(/\/library\/agents\//);

    await waitForAgentPageLoad(page, DETERMINISTIC_AGENT_NAME);
    await expect(
      page
        .locator('a[href*="/library/agents/"]')
        .filter({ hasText: DETERMINISTIC_AGENT_NAME })
        .first(),
    ).toBeVisible({
      timeout: 15000,
    });
    await expect(page).toHaveURL(/\/library\/agents\/[^/?#]+(?:\?.*)?$/);

    libraryAgentId = getLibraryAgentIdFromUrl(page.url());
    expect(libraryAgentId).toBeTruthy();
  } finally {
    if (libraryAgentId) {
      const deleteResponse = await page.request.delete(
        `/api/proxy/api/library/agents/${libraryAgentId}`,
      );
      expect(deleteResponse.ok()).toBe(true);
    }
  }
});
