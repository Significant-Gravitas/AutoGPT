import { Locator, Page, expect } from "@playwright/test";

export async function isVisible(el: Locator, timeout?: number) {
  await expect(el).toBeVisible(timeout ? { timeout } : undefined);
}

export async function hasAttribute(
  el: Locator,
  label: string,
  val: string | RegExp,
) {
  await expect(el).toHaveAttribute(label, val);
}

export async function hasTextContent(el: Locator, text: string | RegExp) {
  await expect(el).toContainText(text);
}

export async function isHidden(el: Locator, timeout?: number) {
  await expect(el).toBeHidden(timeout ? { timeout } : undefined);
}

export async function hasUrl(
  page: Page,
  url: string | RegExp,
  opts?: { decoded?: boolean },
) {
  if (opts?.decoded) {
    return expect(decodeURIComponent(page.url())).toBe(url);
  }

  await expect(page).toHaveURL(url);
}

export async function hasFieldValue(el: Locator, value: string | RegExp) {
  await expect(el).toHaveValue(value);
}

export async function isDisabled(el: Locator) {
  await expect(el).toBeDisabled();
}

export async function isEnabled(el: Locator) {
  await expect(el).toBeEnabled();
}

export async function hasMinCount(el: Locator, minCount: number) {
  const count = await el.count();
  expect(count).toBeGreaterThanOrEqual(minCount);
}

export async function matchesUrl(page: Page, pattern: RegExp) {
  expect(page.url()).toMatch(pattern);
}
