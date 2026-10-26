import { Locator, expect } from "@playwright/test";

export async function isVisible(el: Locator, timeout?: number) {
  await expect(el).toBeVisible(timeout ? { timeout } : undefined);
}
