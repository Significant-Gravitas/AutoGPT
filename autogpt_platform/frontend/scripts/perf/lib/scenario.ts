import type { BrowserContext, Page } from "@playwright/test";
import type { StreamServer } from "./stream-server";

export interface ScenarioContext {
  page: Page;
  context: BrowserContext;
  baseURL: string;
  userID: string;
  stream: StreamServer;
}

export interface ScenarioRun {
  /** The measured interaction. The window stays open `windowMs` after it. */
  run(): Promise<void>;
}

export interface Scenario {
  name: string;
  description: string;
  /** What setup waits for before the window opens, so runs are comparable. */
  readyWhen: string;
  windowMs: number;
  setup(ctx: ScenarioContext): Promise<ScenarioRun>;
}

const NETWORK_IDLE_TIMEOUT_MS = 15_000;
const SETTLE_MS = 1_000;

/** Network idle (or 15 s of polling, whichever is first), then 1 s more. */
export async function settle(page: Page) {
  await page
    .waitForLoadState("networkidle", { timeout: NETWORK_IDLE_TIMEOUT_MS })
    .catch(() => undefined);
  await page.waitForTimeout(SETTLE_MS);
}

export const noInteraction: ScenarioRun = { run: async () => undefined };
