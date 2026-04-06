import { expect, test } from "../coverage-fixture";
import { SMOKE_AUTH_STATES } from "../credentials/accounts";

test.use({ storageState: SMOKE_AUTH_STATES.settings });

test("@smoke settings flow: user can change notification preferences and persist them", async ({
  page,
}) => {
  test.setTimeout(90000);

  await page.goto("/profile/settings");
  await expect(
    page.getByRole("heading", { name: "Notifications", exact: true }),
  ).toBeVisible();

  const firstSwitch = page.getByRole("switch").first();
  const originalState = await firstSwitch.isChecked();

  await firstSwitch.click();
  await expect(
    page.getByRole("button", { name: "Save preferences" }),
  ).toBeEnabled();
  await page.getByRole("button", { name: "Save preferences" }).click();

  await expect(
    page.getByText("Successfully updated notification preferences"),
  ).toBeVisible({ timeout: 15000 });

  await page.reload();
  await expect
    .poll(() => page.getByRole("switch").first().isChecked(), {
      timeout: 10000,
    })
    .toBe(!originalState);
});
