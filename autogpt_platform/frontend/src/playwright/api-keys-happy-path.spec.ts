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

  await page.goto("/settings/api-keys");
  await expect(page).toHaveURL(/\/settings\/api-keys/);
  await expect(
    page.getByText(
      "Manage API keys that let external tools access your AutoGPT account.",
    ),
  ).toBeVisible();

  // The header renders a compact and a full-size "Create Key" button and hides
  // one per breakpoint, so match on the one actually rendered.
  await page
    .getByRole("button", { name: "Create Key" })
    .filter({ visible: true })
    .click();

  const createDialog = page.getByRole("dialog", { name: "Create API key" });
  await createDialog.getByLabel("Name", { exact: true }).fill(keyName);
  const executeGraphCheckbox = createDialog.getByRole("checkbox", {
    name: "Execute Graph",
  });
  const executeGraphChecked =
    (await executeGraphCheckbox.getAttribute("aria-checked")) === "true";
  if (!executeGraphChecked) {
    await executeGraphCheckbox.click();
  }
  await expect(executeGraphCheckbox).toHaveAttribute("aria-checked", "true");

  await createDialog.getByRole("button", { name: "Create Key" }).click();

  const secretDialog = page.getByRole("dialog", { name: "Your new API key" });
  await expect
    .poll(
      async () => {
        if (await secretDialog.isVisible().catch(() => false)) {
          return "created";
        }

        const creationFailed = await page
          .getByText("Failed to create API key")
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

  await secretDialog.getByRole("button", { name: "Copy" }).click();
  await expect(page.getByText("Copied to clipboard")).toBeVisible({
    timeout: 15000,
  });
  await expect
    .poll(() => page.evaluate(() => navigator.clipboard.readText()), {
      timeout: 10000,
    })
    .toBe(createdSecret);

  // Both the header "X" and the footer button are labelled "Close"; either one
  // closes the dialog and resets the form.
  await secretDialog.getByRole("button", { name: "Close" }).first().click();

  const deleteCreatedKeyButton = page.getByRole("button", {
    name: `Delete ${keyName}`,
  });
  await expect(deleteCreatedKeyButton).toBeVisible({ timeout: 15000 });

  await deleteCreatedKeyButton.click();
  const revokeDialog = page.getByRole("dialog", { name: "Revoke API key?" });
  await revokeDialog.getByRole("button", { name: "Revoke key" }).click();

  await expect(page.getByText("API key revoked")).toBeVisible({
    timeout: 15000,
  });
  await expect(deleteCreatedKeyButton).toHaveCount(0);
});
