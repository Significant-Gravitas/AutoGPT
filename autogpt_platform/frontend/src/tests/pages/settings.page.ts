import { expect, Locator, Page } from "@playwright/test";
import { BasePage } from "./base.page";
import { getSelectors } from "../utils/selectors";

export async function getSwitchState(toggle: Locator): Promise<boolean> {
  const ariaChecked = await toggle.getAttribute("aria-checked");
  if (ariaChecked === "true") return true;
  if (ariaChecked === "false") return false;

  const dataState = await toggle.getAttribute("data-state");
  if (dataState === "checked") return true;
  if (dataState === "unchecked") return false;

  const dataChecked = await toggle.getAttribute("data-checked");
  return dataChecked !== null;
}

type ToggleId =
  | "settings-notify-on-agent-run"
  | "settings-notify-on-block-execution-failed"
  | "settings-notify-on-continuous-agent-error"
  | "settings-notify-on-zero-balance"
  | "settings-notify-on-low-balance"
  | "settings-notify-on-daily-summary"
  | "settings-notify-on-weekly-summary"
  | "settings-notify-on-monthly-summary";

export const TOGGLE_IDS: ReadonlyArray<ToggleId> = [
  "settings-notify-on-agent-run",
  "settings-notify-on-block-execution-failed",
  "settings-notify-on-continuous-agent-error",
  "settings-notify-on-zero-balance",
  "settings-notify-on-low-balance",
  "settings-notify-on-daily-summary",
  "settings-notify-on-weekly-summary",
  "settings-notify-on-monthly-summary",
];

export class SettingsPage extends BasePage {
  static readonly path = "/profile/settings";

  constructor(page: Page) {
    super(page);
  }

  async goto(): Promise<void> {
    await this.page.goto(SettingsPage.path);
    await this.isLoaded();
  }

  async isLoaded(): Promise<boolean> {
    try {
      await this.page.waitForLoadState("domcontentloaded");
      const header = this.page.getByRole("heading", { name: "My account" });
      const email = this.getEmailInput();

      await Promise.all([
        header.waitFor({ state: "visible" }),
        email.waitFor({ state: "visible" }),
      ]);

      return true;
    } catch {
      return false;
    }
  }

  getEmailInput(): Locator {
    return this.page.getByTestId("settings-email");
  }
  getPasswordInput(): Locator {
    return this.page.getByTestId("settings-password");
  }
  getConfirmPasswordInput(): Locator {
    return this.page.getByTestId("settings-confirm-password");
  }
  getToggle(id: ToggleId): Locator {
    return this.page.getByTestId(id);
  }
  getCancelButton(): Locator {
    return this.page.getByTestId("settings-cancel");
  }
  getSaveButton(): Locator {
    return this.page.getByRole("button", { name: /Save changes|Saving\.\.\./ });
  }

  async setEmail(value: string): Promise<void> {
    const input = this.getEmailInput();
    await input.waitFor({ state: "visible" });
    await input.fill(value);
    await expect(input).toHaveValue(value);
  }

  async setPassword(value: string): Promise<void> {
    const input = this.getPasswordInput();
    await input.waitFor({ state: "visible" });
    await input.fill(value);
    await expect(input).toHaveValue(value);
  }

  async setConfirmPassword(value: string): Promise<void> {
    const input = this.getConfirmPasswordInput();
    await input.waitFor({ state: "visible" });
    await input.fill(value);
    await expect(input).toHaveValue(value);
  }

  private async getSwitchState(toggle: Locator): Promise<boolean> {
    const ariaChecked = await toggle.getAttribute("aria-checked");
    if (ariaChecked === "true") return true;
    if (ariaChecked === "false") return false;

    const dataState = await toggle.getAttribute("data-state");
    if (dataState === "checked") return true;
    if (dataState === "unchecked") return false;

    const dataChecked = await toggle.getAttribute("data-checked");
    return dataChecked !== null;
  }

  private async setSwitchState(toggle: Locator, desired: boolean): Promise<void> {
    await toggle.waitFor({ state: "visible" });
    const current = await this.getSwitchState(toggle);
    if (current !== desired) {
      await toggle.click();
      await expect
        .poll(async () => this.getSwitchState(toggle))
        .toBe(desired);
    }
  }

  async toggle(id: ToggleId): Promise<void> {
    await this.getToggle(id).click();
  }

  async enable(id: ToggleId): Promise<void> {
    await this.setSwitchState(this.getToggle(id), true);
  }

  async disable(id: ToggleId): Promise<void> {
    await this.setSwitchState(this.getToggle(id), false);
  }

  async cancelChanges(): Promise<void> {
    const btn = this.getCancelButton();
    await btn.waitFor({ state: "visible" });
    await btn.click();
    await this.getEmailInput().waitFor({ state: "visible" });
  }

  async saveChanges(): Promise<void> {
    const btn = this.getSaveButton();
    await btn.waitFor({ state: "visible" });
    await btn.click();
    await this.waitForSaveComplete();
  }

  async waitForSaveComplete(): Promise<void> {
    const { getText } = getSelectors(this.page);
    const toast = getText("Successfully updated settings");

    await Promise.race([
      toast.waitFor({ state: "visible" }),
      expect(this.getSaveButton()).toHaveText("Save changes"),
    ]);
  }

  async getEmailValue(): Promise<string> {
    return this.getEmailInput().inputValue();
  }

  async expectValidationError(text: string | RegExp): Promise<void> {
    const { getText } = getSelectors(this.page);
    await expect(getText(text)).toBeVisible();
  }
}

export async function navigateToSettings(page: Page): Promise<SettingsPage> {
  const settings = new SettingsPage(page);
  await settings.goto();
  return settings;
}
