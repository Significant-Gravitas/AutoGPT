import { Locator, Page } from "@playwright/test";
import { BasePage } from "./base.page";
import { getSelectors } from "../utils/selectors";

export class ProfileFormPage extends BasePage {
  constructor(page: Page) {
    super(page);
  }

  private getId(id: string | RegExp): Locator {
    const { getId } = getSelectors(this.page);
    return getId(id);
  }

  private async hideFloatingWidgets(): Promise<void> {
    await this.page.addStyleTag({
      content: `
        [data-tally-open] { display: none !important; }
      `,
    });
  }

  // Locators
  title(): Locator {
    return this.getId("profile-info-form-title");
  }

  displayNameField(): Locator {
    const { getField } = getSelectors(this.page);
    return getField("Display name");
  }

  handleField(): Locator {
    const { getField } = getSelectors(this.page);
    return getField("Handle");
  }

  bioField(): Locator {
    const { getField } = getSelectors(this.page);
    return getField("Bio");
  }

  linkField(index: number): Locator {
    this.assertValidLinkIndex(index);
    const { getField } = getSelectors(this.page);
    return getField(`Link ${index}`);
  }

  cancelButton(): Locator {
    const { getButton } = getSelectors(this.page);
    return getButton("Cancel");
  }

  saveButton(): Locator {
    const { getButton } = getSelectors(this.page);
    return getButton("Save changes");
  }

  // State
  async isLoaded(): Promise<boolean> {
    try {
      await this.title().waitFor({ state: "visible", timeout: 10_000 });
      await this.displayNameField().waitFor({
        state: "visible",
        timeout: 10_000,
      });
      await this.handleField().waitFor({ state: "visible", timeout: 10_000 });
      return true;
    } catch {
      return false;
    }
  }

  // Actions
  async setDisplayName(name: string): Promise<void> {
    await this.displayNameField().fill(name);
  }

  async getDisplayName(): Promise<string> {
    return this.displayNameField().inputValue();
  }

  async setHandle(handle: string): Promise<void> {
    await this.handleField().fill(handle);
  }

  async getHandle(): Promise<string> {
    return this.handleField().inputValue();
  }

  async setBio(bio: string): Promise<void> {
    await this.bioField().fill(bio);
  }

  async getBio(): Promise<string> {
    return this.bioField().inputValue();
  }

  async setLink(index: number, url: string): Promise<void> {
    await this.linkField(index).fill(url);
  }

  async getLink(index: number): Promise<string> {
    return this.linkField(index).inputValue();
  }

  async setLinks(links: Array<string | undefined>): Promise<void> {
    for (let i = 1; i <= 5; i++) {
      const val = links[i - 1] ?? "";
      await this.setLink(i, val);
    }
  }

  async clickCancel(): Promise<void> {
    await this.cancelButton().click();
  }

  async clickSave(): Promise<void> {
    await this.saveButton().click();
  }

  async saveChanges(): Promise<void> {
    await this.hideFloatingWidgets();
    await this.clickSave();
    await this.waitForSaveComplete();
  }

  async waitForSaveComplete(timeoutMs: number = 15_000): Promise<void> {
    const { getButton } = getSelectors(this.page);
    await getButton("Save changes").waitFor({
      state: "attached",
      timeout: timeoutMs,
    });
    await getButton("Save changes").waitFor({
      state: "visible",
      timeout: timeoutMs,
    });
  }

  private assertValidLinkIndex(index: number) {
    if (index < 1 || index > 5) {
      throw new Error(`Link index must be between 1 and 5. Received: ${index}`);
    }
  }
}
