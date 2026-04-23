import { readFileSync } from "fs";
import { dirname, resolve } from "path";
import { fileURLToPath } from "url";
import { runInNewContext } from "vm";
import { describe, expect, it } from "vitest";

/**
 * Loads public/push-sw.js into a sandbox and exposes the helpers it defines
 * as top-level vars. The SW file has to be hand-written JS (Next.js serves
 * public/ as-is), so these tests are our only guarantee that the routing
 * and suppression logic stays correct.
 */
function loadServiceWorkerHelpers() {
  const here = dirname(fileURLToPath(import.meta.url));
  const swPath = resolve(here, "../../../public/push-sw.js");
  const source = readFileSync(swPath, "utf8");

  // Stub globals the SW touches during evaluation — we only need the helper
  // functions, not the install/push/click listeners (those no-op in this
  // sandbox).
  const sandbox: Record<string, unknown> = {
    self: {
      skipWaiting: () => undefined,
      clients: { claim: () => Promise.resolve(), matchAll: () => [] },
      addEventListener: () => undefined,
      location: { origin: "https://example.com" },
      registration: { pushManager: {} },
    },
    URL,
    Array,
  };
  runInNewContext(source, sandbox);
  return sandbox as {
    NOTIFICATION_MAP: Record<
      string,
      Record<string, { title: string; body: string; url: string }>
    >;
    getNotificationConfig: (data: { type?: string; event?: string }) => {
      title: string;
      body: string;
      url: string;
    };
    isClientViewingTarget: (
      client: { visibilityState: string; focused: boolean; url: string },
      targetUrl: string,
    ) => boolean;
  };
}

describe("push-sw getNotificationConfig", () => {
  const sw = loadServiceWorkerHelpers();

  it("returns copilot session_completed config", () => {
    const config = sw.getNotificationConfig({
      type: "copilot_completion",
      event: "session_completed",
    });
    expect(config.title).toBe("AutoPilot is ready");
    expect(config.url).toBe("/copilot");
  });

  it("returns copilot session_failed config", () => {
    const config = sw.getNotificationConfig({
      type: "copilot_completion",
      event: "session_failed",
    });
    expect(config.title).toBe("AutoPilot session failed");
  });

  it("falls back to generic notification for unknown type", () => {
    const config = sw.getNotificationConfig({
      type: "unknown_type",
      event: "anything",
    });
    expect(config.title).toBe("AutoGPT Notification");
    expect(config.url).toBe("/");
  });

  it("falls back when event is unknown within a known type", () => {
    const config = sw.getNotificationConfig({
      type: "copilot_completion",
      event: "never_seen_event",
    });
    expect(config.title).toBe("AutoGPT Notification");
  });
});

describe("push-sw isClientViewingTarget", () => {
  const sw = loadServiceWorkerHelpers();

  function client(args: {
    url: string;
    visibilityState?: string;
    focused?: boolean;
  }) {
    return {
      visibilityState: args.visibilityState ?? "visible",
      focused: args.focused ?? true,
      url: args.url,
    };
  }

  it("returns false when tab is not visible", () => {
    const result = sw.isClientViewingTarget(
      client({ url: "https://example.com/copilot", visibilityState: "hidden" }),
      "/copilot",
    );
    expect(result).toBe(false);
  });

  it("returns false when window is not focused", () => {
    const result = sw.isClientViewingTarget(
      client({ url: "https://example.com/copilot", focused: false }),
      "/copilot",
    );
    expect(result).toBe(false);
  });

  it("returns true when pathname matches and no query params expected", () => {
    const result = sw.isClientViewingTarget(
      client({ url: "https://example.com/copilot" }),
      "/copilot",
    );
    expect(result).toBe(true);
  });

  it("returns false when pathname differs", () => {
    const result = sw.isClientViewingTarget(
      client({ url: "https://example.com/library" }),
      "/copilot",
    );
    expect(result).toBe(false);
  });

  it("returns true when sessionId matches", () => {
    const result = sw.isClientViewingTarget(
      client({ url: "https://example.com/copilot?sessionId=A" }),
      "/copilot?sessionId=A",
    );
    expect(result).toBe(true);
  });

  it("returns false when sessionId differs — prevents wrong-session suppression", () => {
    const result = sw.isClientViewingTarget(
      client({ url: "https://example.com/copilot?sessionId=A" }),
      "/copilot?sessionId=B",
    );
    expect(result).toBe(false);
  });

  it("ignores extra client params the target doesn't care about", () => {
    const result = sw.isClientViewingTarget(
      client({ url: "https://example.com/copilot?utm_source=email" }),
      "/copilot",
    );
    expect(result).toBe(true);
  });
});
