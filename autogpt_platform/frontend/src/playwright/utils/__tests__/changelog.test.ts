import type { BrowserContext, Route } from "@playwright/test";
import { readFileSync } from "node:fs";
import path from "node:path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { STORAGE_KEY } from "../../../components/molecules/ChangelogPopup/changelog-constants";
import { markChangelogSeen } from "../changelog";

async function requestChangelog(markdown?: string) {
  let handler: ((route: Route) => Promise<void>) | undefined;
  const context = {
    async route(_url: string, callback: typeof handler) {
      handler = callback;
    },
  } as unknown as BrowserContext;
  const upstream = { ok: () => false, status: () => 502 };
  const fetch = vi.fn(async () => upstream);
  const fulfill = vi.fn(async () => {});
  const route = {
    fetch,
    fulfill,
    request: () => ({
      frame: () => ({
        evaluate: async (callback: (args: string[]) => void, args: string[]) =>
          callback(args),
      }),
    }),
  } as unknown as Route;
  await markChangelogSeen(context, markdown);
  expect(handler).toBeDefined();
  await handler!(route);
  return { fetch, fulfill, upstream };
}

beforeEach(() => window.localStorage.clear());

describe("Playwright changelog input", () => {
  it("serves the local fixture and marks it seen without contacting the publisher", async () => {
    const markdown = readFileSync(
      path.resolve(__dirname, "../../fixtures/changelog.md"),
      "utf8",
    );
    const { fetch, fulfill } = await requestChangelog(markdown);
    expect(fetch).not.toHaveBeenCalled();
    expect(fulfill).toHaveBeenCalledWith({
      status: 200,
      contentType: "text/markdown; charset=utf-8",
      body: markdown,
    });
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe(
      "january-1-january-2-2026",
    );
  });

  it("preserves upstream failures when no fixture was requested", async () => {
    const { fetch, fulfill, upstream } = await requestChangelog();
    expect(fetch).toHaveBeenCalledOnce();
    expect(fulfill).toHaveBeenCalledWith({ response: upstream });
    expect(window.localStorage.getItem(STORAGE_KEY)).toBeNull();
  });
});
