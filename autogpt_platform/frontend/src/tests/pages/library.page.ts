import { Locator, Page } from "@playwright/test";
import { getSelectors } from "../utils/selectors";

// Locator functions
export function getLibraryTab(page: Page): Locator {
  return page.locator('a[href="/library"]');
}

export function getAgentCards(page: Page): Locator {
  return page.getByTestId("library-agent-card");
}

export function getNewRunButton(page: Page): Locator {
  return page.getByRole("button", { name: "New run" });
}

export function getAgentTitle(page: Page): Locator {
  return page.locator("h1").first();
}

// Action functions
export async function navigateToLibrary(page: Page): Promise<void> {
  await getLibraryTab(page).click();
  await page.waitForURL(/.*\/library/);
}

export async function clickFirstAgent(page: Page): Promise<void> {
  const firstAgent = getAgentCards(page).first();
  await firstAgent.click();
}

export async function navigateToAgentByName(
  page: Page,
  agentName: string,
): Promise<void> {
  const agentCard = getAgentCards(page).filter({ hasText: agentName }).first();
  await agentCard.click();
}

export async function clickRunButton(page: Page): Promise<void> {
  const { getId } = getSelectors(page);
  const runButton = getId("agent-run-button");
  const runAgainButton = getId("run-again-button");

  if (await runButton.isVisible()) {
    await runButton.click();
  } else if (await runAgainButton.isVisible()) {
    await runAgainButton.click();
  } else {
    throw new Error("Neither run button nor run again button is visible");
  }
}

export async function clickNewRunButton(page: Page): Promise<void> {
  await getNewRunButton(page).click();
}

export async function runAgent(page: Page): Promise<void> {
  await clickRunButton(page);
}

export async function waitForAgentPageLoad(page: Page): Promise<void> {
  await page.waitForURL(/.*\/library\/agents\/[^/]+/);
  await page.getByTestId("Run actions").isVisible({ timeout: 10000 });
}

export async function getAgentName(page: Page): Promise<string> {
  return (await getAgentTitle(page).textContent()) || "";
}

export async function isLoaded(page: Page): Promise<boolean> {
  return await page.locator("h1").isVisible();
}

export async function waitForRunToComplete(
  page: Page,
  timeout = 30000,
): Promise<void> {
  await page.waitForSelector(".bg-green-500, .bg-red-500, .bg-purple-500", {
    timeout,
  });
}

export async function getRunStatus(page: Page): Promise<string> {
  if (await page.locator(".animate-spin").isVisible()) {
    return "running";
  } else if (await page.locator(".bg-green-500").isVisible()) {
    return "completed";
  } else if (await page.locator(".bg-red-500").isVisible()) {
    return "failed";
  }
  return "unknown";
}
