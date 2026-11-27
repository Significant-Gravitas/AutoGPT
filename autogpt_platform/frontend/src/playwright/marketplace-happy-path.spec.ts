import { expect, test } from "./coverage-fixture";
import { E2E_AUTH_STATES } from "./credentials/accounts";
import {
  clickRunButton,
  dismissFeedbackDialog,
  LibraryPage,
} from "./pages/library.page";
import { MarketplacePage } from "./pages/marketplace.page";

test.use({ storageState: E2E_AUTH_STATES.marketplace });

const MISSING_SKILL_SLUG = "e2e-no-such-skill";

test("marketplace happy path: a missing skill shows the 404 page, not an application error", async ({
  page,
}) => {
  test.setTimeout(90000);

  // notFound() during an in-flight Server Action must still let the router's
  // next update render (#17278). Every Server Action is held until the 404
  // page shows, so notFound() runs while one is in flight. React holds that
  // 404 back until the page renders again, which the session check's 10 s
  // deadline (SESSION_CHECK_DEADLINE_MS) forces while its action is held.
  let heldServerActions = 0;
  let releaseServerActions = () => {};
  const serverActionsReleased = new Promise<void>((resolve) => {
    releaseServerActions = resolve;
  });
  await page.route(
    `**/marketplace/skills/${MISSING_SKILL_SLUG}`,
    async (route) => {
      if (route.request().headers()["next-action"]) {
        heldServerActions += 1;
        await serverActionsReleased;
      }
      await route.fallback();
    },
  );
  const notFoundPage = page.getByText("This page could not be found.");

  try {
    await page.goto(`/marketplace/skills/${MISSING_SKILL_SLUG}`);
    await expect(notFoundPage).toBeVisible({ timeout: 30000 });
    expect(heldServerActions).toBeGreaterThan(0);
  } finally {
    releaseServerActions();
  }

  await expect(page.getByText("Application error")).toHaveCount(0);
  await expect(notFoundPage).toBeVisible();
  // A client-side navigation is the router's next update.
  await page
    .getByRole("link", { name: "Marketplace", exact: true })
    .filter({ visible: true })
    .first()
    .click({ timeout: 15000 });
  await expect(page.getByTestId("store-search-input")).toBeVisible({
    timeout: 20000,
  });
});

test("marketplace happy path: user can browse Marketplace and open an agent detail page", async ({
  page,
}) => {
  test.setTimeout(90000);

  const marketplacePage = new MarketplacePage(page);
  await marketplacePage.openFeaturedAgent();

  await expect(page.getByTestId("agent-description")).toBeVisible();
});

test("marketplace happy path: user can add a Marketplace agent to Library and run it", async ({
  page,
}) => {
  test.setTimeout(120000);

  const marketplacePage = new MarketplacePage(page);
  await marketplacePage.openRunnableAgent();

  const agentName = await page.getByTestId("agent-title").innerText();

  await Promise.all([
    expect(page.getByText("Redirecting to your library...")).toBeVisible(),
    page.getByTestId("agent-add-library-button").click(),
  ]);
  await expect(page).toHaveURL(/\/library\/agents\//);

  const libraryPage = new LibraryPage(page);
  await libraryPage.openSavedAgent(agentName);
  await clickRunButton(page);

  await libraryPage.waitForRunToComplete();
  await dismissFeedbackDialog(page);

  const runStatus = await libraryPage.getRunStatus();
  expect(runStatus).toBe("completed");
  // The seeded listing adds 34 to its input, which defaults to 8.
  await libraryPage.assertRunOutputValue("Output", /^42(?:\.0+)?$/);
});
