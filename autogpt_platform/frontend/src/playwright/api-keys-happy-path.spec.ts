import { randomUUID } from "crypto";
import { expect, test } from "./coverage-fixture";
import { E2E_AUTH_STATES } from "./credentials/accounts";

test.use({ storageState: E2E_AUTH_STATES.parallelB });

test("api keys happy path: user can create, copy, and revoke an API key", async ({
  page,
  context,
}) => {
  test.setTimeout(120000);

  await context.grantPermissions(["clipboard-read", "clipboard-write"]);

  const keyName = `E2E CLI Key ${randomUUID().slice(0, 8)}`;

  await page.goto("/profile/api-keys");
  await expect(page).toHaveURL(/\/profile\/api-keys/);
  await expect(
    page.getByText(
      "Manage your AutoGPT Platform API keys for programmatic access",
    ),
  ).toBeVisible();

  await page.getByRole("button", { name: "Create Key" }).click();
  await page.getByLabel("Name").fill(keyName);
  const executeGraphCheckbox = page.getByRole("checkbox", {
    name: /EXECUTE_GRAPH/i,
  });
  const executeGraphChecked =
    (await executeGraphCheckbox.getAttribute("aria-checked")) === "true";
  if (!executeGraphChecked) {
    await executeGraphCheckbox.click();
  }
  await expect(executeGraphCheckbox).toHaveAttribute("aria-checked", "true");

  await page.getByRole("button", { name: "Create" }).click();

  const secretDialog = page.getByRole("dialog", {
    name: "AutoGPT Platform API Key Created",
  });
  await expect
    .poll(
      async () => {
        if (await secretDialog.isVisible().catch(() => false)) {
          return "created";
        }

        const creationFailed = await page
          .getByText("Failed to create AutoGPT Platform API key")
          .isVisible()
          .catch(() => false);
        if (creationFailed) {
          return "failed";
        }

        return "pending";
      },
      {
        timeout: 30000,
        message:
          "API key creation should either open the created-key dialog or surface an explicit failure toast",
      },
    )
    .toBe("created");
  await expect(secretDialog).toBeVisible();

  const createdSecret = (
    (await secretDialog.locator("code").textContent()) ?? ""
  ).trim();
  expect(createdSecret.length).toBeGreaterThan(0);

  await secretDialog.getByRole("button").first().click();
  await expect(page.getByText("Copied", { exact: true })).toBeVisible({
    timeout: 15000,
  });
  await expect
    .poll(() => page.evaluate(() => navigator.clipboard.readText()), {
      timeout: 10000,
    })
    .toBe(createdSecret);

  await secretDialog.getByRole("button", { name: "Close" }).first().click();

  const createdKeyRow = page
    .getByTestId("api-key-row")
    .filter({ hasText: keyName })
    .first();
  await expect(createdKeyRow).toBeVisible({ timeout: 15000 });

  await createdKeyRow.getByTestId("api-key-actions").click();
  await page.getByRole("menuitem", { name: "Revoke" }).click();

  await expect(
    page.getByText("AutoGPT Platform API key revoked successfully"),
  ).toBeVisible({ timeout: 15000 });
  await expect(
    page.getByTestId("api-key-row").filter({ hasText: keyName }),
  ).toHaveCount(0);
});
