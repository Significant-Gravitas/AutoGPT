import { Locator, Page } from "@playwright/test";

// Notification functions - these are part of the header and available on all pages
export function getNotificationButton(page: Page): Locator {
  return page.locator('button[title="Agent Activity"]');
}

export function getNotificationBadge(page: Page): Locator {
  return page.locator('button[title="Agent Activity"] .animate-spin').first();
}

export function getNotificationDropdown(page: Page): Locator {
  return page.locator('[role="dialog"]:has-text("Agent Activity")');
}

export function getNotificationItemsLocator(page: Page): Locator {
  return getNotificationDropdown(page).locator('[role="button"]');
}

export async function clickNotificationButton(page: Page): Promise<void> {
  await getNotificationButton(page).click();
}

export async function isNotificationBadgeVisible(page: Page): Promise<boolean> {
  return await getNotificationBadge(page).isVisible();
}

export async function isNotificationDropdownVisible(
  page: Page,
): Promise<boolean> {
  return await getNotificationDropdown(page).isVisible();
}

export async function getNotificationCount(page: Page): Promise<string> {
  const badge = page.locator('button[title="Agent Activity"] .bg-purple-600');
  return (await badge.textContent()) || "0";
}

export async function getNotificationItems(
  page: Page,
): Promise<{ name: string; status: string; time: string }[]> {
  const items = await getNotificationItemsLocator(page).all();
  const results = [];

  for (const item of items) {
    const name = (await item.locator(".truncate").textContent()) || "";
    const time = (await item.locator(".\\!text-zinc-500").textContent()) || "";

    // Determine status from icon classes and text content
    let status = "unknown";
    if (await item.locator(".animate-spin").isVisible()) {
      status = "running";
    } else if (await item.locator("svg").first().isVisible()) {
      // For non-animated icons, check the text content to determine status
      const timeText = time.toLowerCase();
      if (timeText.includes("completed")) {
        status = "completed";
      } else if (timeText.includes("failed")) {
        status = "failed";
      } else if (timeText.includes("stopped")) {
        status = "terminated";
      } else if (timeText.includes("incomplete")) {
        status = "incomplete";
      } else if (timeText.includes("queued")) {
        status = "queued";
      }
    }

    results.push({ name, status, time });
  }

  return results;
}

export async function waitForNotificationUpdate(
  page: Page,
  _timeout = 10000,
): Promise<void> {
  await page.waitForTimeout(_timeout); // Wait for potential updates
}

export async function hasNotificationWithStatus(
  page: Page,
  status: string,
): Promise<boolean> {
  const items = await getNotificationItems(page);
  return items.some((item) => item.status === status);
}

export async function getNotificationByAgentName(
  page: Page,
  agentName: string,
): Promise<{ name: string; status: string; time: string } | null> {
  const items = await getNotificationItems(page);
  return items.find((item) => item.name.includes(agentName)) || null;
}
