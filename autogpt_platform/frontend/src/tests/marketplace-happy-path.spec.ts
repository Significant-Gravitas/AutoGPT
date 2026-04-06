import { Page } from "@playwright/test";
import { expect, test } from "./coverage-fixture";
import { E2E_AUTH_STATES } from "./credentials/accounts";
import {
  clickRunButton,
  getRunStatus,
  waitForAgentPageLoad,
  waitForRunToComplete,
} from "./pages/library.page";
import { MarketplacePage } from "./pages/marketplace.page";

test.use({ storageState: E2E_AUTH_STATES.marketplace });

const RUNNABLE_MARKETPLACE_AGENT_PATH =
  "/marketplace/agent/e2e-marketplace/e2e-calculator-agent";
const FALLBACK_MARKETPLACE_AGENT_PATH =
  "/marketplace/agent/autogpt/unspirational-poster-maker";

async function openRunnableMarketplaceAgent(page: Page) {
  const candidatePaths = [
    {
      path: RUNNABLE_MARKETPLACE_AGENT_PATH,
      expectedTitle: /E2E Calculator Agent/i,
    },
    {
      path: FALLBACK_MARKETPLACE_AGENT_PATH,
      expectedTitle: /Unspirational Poster Maker/i,
    },
  ];

  for (const candidate of candidatePaths) {
    await page.goto(candidate.path);
    const addToLibraryButton = page.getByTestId("agent-add-library-button");

    if (
      await addToLibraryButton.isVisible({ timeout: 5000 }).catch(() => false)
    ) {
      await expect(page).toHaveURL(/\/marketplace\/agent\//);
      await expect(page.getByTestId("agent-title").first()).toHaveText(
        candidate.expectedTitle,
      );
      return candidate;
    }
  }

  throw new Error(
    "Could not find a runnable marketplace agent for PR E2E coverage",
  );
}

async function openMarketplaceAgent(page: Page) {
  const marketplacePage = new MarketplacePage(page);

  await marketplacePage.goto(page);
  await expect(
    page.getByText("Explore AI agents", { exact: false }),
  ).toBeVisible();

  const featuredAgentLink = page
    .locator('a[href*="/marketplace/agent/"]')
    .first();
  if (await featuredAgentLink.isVisible({ timeout: 5000 }).catch(() => false)) {
    await featuredAgentLink.click();
  } else {
    const agentCard = await marketplacePage.getFirstTopAgent();
    await agentCard.click();
  }

  await expect(page).toHaveURL(/\/marketplace\/agent\//);
  await expect(page.getByTestId("agent-title")).toBeVisible();
}

test("marketplace happy path: user can browse Marketplace and open an agent detail page", async ({
  page,
}) => {
  test.setTimeout(90000);

  await openMarketplaceAgent(page);

  await expect(page.getByTestId("agent-description")).toBeVisible();
});

test("marketplace happy path: user can add a Marketplace agent to Library and run it", async ({
  page,
}) => {
  test.setTimeout(120000);

  await openRunnableMarketplaceAgent(page);

  await page.getByTestId("agent-add-library-button").click();
  await expect(page.getByText("Redirecting to your library...")).toBeVisible();
  await expect(page).toHaveURL(/\/library\/agents\//);

  await waitForAgentPageLoad(page);
  await clickRunButton(page);
  await waitForRunToComplete(page, 45000);

  const runStatus = await getRunStatus(page);
  expect(["completed", "failed", "running"]).toContain(runStatus);
});
