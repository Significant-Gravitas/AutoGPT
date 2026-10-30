// Drive the app's own window. The app is started the way a user starts it,
// with one extra switch that opens Chromium's debugging port; Playwright then
// attaches to the running app instead of launching a browser of its own.

import net from "node:net";

import { chromium, type Browser, type Page } from "@playwright/test";

const POLL_MS = 500;

export interface Attached {
  browser: Browser;
  /** The main window, which shows the app at its public URL. */
  page: Page;
}

export function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address() as net.AddressInfo;
      server.close(() => resolve(port));
    });
  });
}

export async function attach(debugPort: number, appUrl: string, timeoutMs: number): Promise<Attached> {
  const endpoint = `http://127.0.0.1:${debugPort}`;
  const deadline = Date.now() + timeoutMs;
  await poll(deadline, `the debugging port ${debugPort} to answer`, async () => {
    const response = await fetch(`${endpoint}/json/version`);
    return response.ok;
  });
  const browser = await chromium.connectOverCDP(endpoint);
  const origin = new URL(appUrl).origin;
  let page: Page | undefined;
  // The main window opens a moment after `ready`; the startup window (a
  // file:// page) is there from the beginning.
  await poll(deadline, `a window showing ${origin}`, async () => {
    page = browser
      .contexts()
      .flatMap((context) => context.pages())
      .find((candidate) => candidate.url().startsWith(origin));
    return page !== undefined;
  });
  return { browser, page: page as Page };
}

async function poll(deadline: number, what: string, check: () => Promise<boolean>): Promise<void> {
  for (;;) {
    try {
      if (await check()) return;
    } catch {
      // Not there yet.
    }
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${what}`);
    await new Promise((resolve) => setTimeout(resolve, POLL_MS));
  }
}
