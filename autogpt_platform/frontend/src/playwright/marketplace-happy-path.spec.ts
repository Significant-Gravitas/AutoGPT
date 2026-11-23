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

  // notFound() while a Server Action was in flight left the router with a
  // truncated hook list, so its next update replaced the whole app with
  // "Application error" (#17278). Holding every Server Action until the
  // skill read gives up puts notFound() in that window on every run.
  let releaseServerActions = () => {};
  const serverActionsReleased = new Promise<void>((resolve) => {
    releaseServerActions = resolve;
  });
  await page.route(
    `**/marketplace/skills/${MISSING_SKILL_SLUG}`,
    async (route) => {
      if (route.request().headers()["next-action"]) {
        await serverActionsReleased;
      }
      await route.fallback();
    },
  );
  let failedReads = 0;
  page.on("requestfinished", async (request) => {
    if (!request.url().endsWith(`/store/skills/${MISSING_SKILL_SLUG}`)) return;
    if ((await request.response())?.status() === 404) failedReads += 1;
  });

  await page.goto(`/marketplace/skills/${MISSING_SKILL_SLUG}`);
  // The first read and React Query's three retries.
  await expect.poll(() => failedReads, { timeout: 30000 }).toBe(4);
  releaseServerActions();

  await expect(page.getByText("This page could not be found.")).toBeVisible({
    timeout: 15000,
  });
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
