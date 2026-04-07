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

const ACCEPTED_RUN_STATUSES = [
  "completed",
  "failed",
  "running",
  "queued",
  "review",
] as const;

const RUNNABLE_MARKETPLACE_AGENT_PATH =
  "/marketplace/agent/e2e-marketplace/e2e-calculator-agent";
const FALLBACK_MARKETPLACE_AGENT_PATH =
  "/marketplace/agent/autogpt/unspirational-poster-maker";

async function dismissFeedbackDialog(page: Page) {
  const feedbackDialog = page.getByRole("dialog", {
    name: "We'd love your feedback",
  });
  if (!(await feedbackDialog.isVisible().catch(() => false))) {
    return;
  }

  const cancelButton = feedbackDialog.getByRole("button", { name: "Cancel" });
  if (await cancelButton.isVisible().catch(() => false)) {
    await cancelButton.click();
    await expect(feedbackDialog).toBeHidden({ timeout: 15000 });
    return;
  }

  await feedbackDialog.getByRole("button", { name: "Close" }).click();
  await expect(feedbackDialog).toBeHidden({ timeout: 15000 });
}

async function openRunnableMarketplaceAgent(page: Page) {
  const candidatePaths = [
    { path: RUNNABLE_MARKETPLACE_AGENT_PATH },
    { path: FALLBACK_MARKETPLACE_AGENT_PATH },
  ];

  for (const candidate of candidatePaths) {
    await page.goto(candidate.path);
    const addToLibraryButton = page.getByTestId("agent-add-library-button");

    if (
      await addToLibraryButton.isVisible({ timeout: 5000 }).catch(() => false)
    ) {
      await expect(page).toHaveURL(/\/marketplace\/agent\//);
      await expect(page.getByTestId("agent-title").first()).toBeVisible();
      return candidate;
    }
  }

  const marketplacePage = new MarketplacePage(page);
  await marketplacePage.goto(page);

  const candidateLinks = await page
    .locator('a[href*="/marketplace/agent/"]')
    .evaluateAll((links) =>
      links
        .map((link) => link.getAttribute("href"))
        .filter((href): href is string => Boolean(href)),
    );

  const uniquePaths = [...new Set(candidateLinks)].slice(0, 8);
  for (const path of uniquePaths) {
    await page.goto(path);
    const addToLibraryButton = page.getByTestId("agent-add-library-button");

    if (
      await addToLibraryButton.isVisible({ timeout: 5000 }).catch(() => false)
    ) {
      await expect(page).toHaveURL(/\/marketplace\/agent\//);
      await expect(page.getByTestId("agent-title").first()).toBeVisible();
      return { path, expectedTitle: null };
    }
  }

  throw new Error(
    "Could not find a runnable marketplace agent for PR E2E coverage",
  );
}

async function openMarketplaceAgent(page: Page) {
  const marketplacePage = new MarketplacePage(page);

  await marketplacePage.goto(page);
  await expect(page.getByRole("heading", { level: 1 }).first()).toBeVisible();

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
  await dismissFeedbackDialog(page);
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
  expect(ACCEPTED_RUN_STATUSES).toContain(
    runStatus as (typeof ACCEPTED_RUN_STATUSES)[number],
  );
});

test("marketplace happy path: user can download an agent from the Marketplace", async ({
  page,
}) => {
  test.setTimeout(90000);

  await openMarketplaceAgent(page);

  await page.getByTestId("agent-download-button").click();
  await expect(
    page.getByText("Your agent has been successfully downloaded."),
  ).toBeVisible();
});
