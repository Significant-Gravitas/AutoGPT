import { Locator, Page } from "@playwright/test";

function getNotificationButton(page: Page): Locator {
  return page.locator('button[title="Agent Activity"]');
}

export const headerTest = {
  getNotificationButton,
};
