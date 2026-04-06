import { expect, test } from "../coverage-fixture";
import { SMOKE_AUTH_STATES } from "../credentials/accounts";
import { MarketplacePage } from "../pages/marketplace.page";

test.use({ storageState: SMOKE_AUTH_STATES.marketplace });

test("@smoke marketplace flow: user can browse marketplace agents, download one, and add one to the library", async ({
  page,
}) => {
  test.setTimeout(90000);

  const marketplacePage = new MarketplacePage(page);

  await marketplacePage.goto(page);
  await expect(
    page.getByText("Explore AI agents", { exact: false }),
  ).toBeVisible();

  const firstStoreCard = await marketplacePage.getFirstTopAgent();
  await firstStoreCard.click();

  await expect(page).toHaveURL(/\/marketplace\/agent\//);
  await expect(page.getByTestId("agent-title")).toBeVisible();

  await page.getByTestId("agent-download-button").click();
  await expect(
    page.getByText("Your agent has been successfully downloaded."),
  ).toBeVisible();

  const agentName = (
    await page.getByTestId("agent-title").textContent()
  )?.trim();
  expect(agentName).toBeTruthy();

  await page.getByTestId("agent-add-library-button").click();
  await expect(page.getByText("Redirecting to your library...")).toBeVisible();
  await expect(page).toHaveURL(/\/library\/agents\//);

  if (agentName) {
    await expect(page).toHaveTitle(
      new RegExp(
        `${agentName.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")} - Library - AutoGPT Platform`,
      ),
    );
  }
});
