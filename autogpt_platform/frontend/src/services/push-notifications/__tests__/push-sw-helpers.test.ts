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
  const swPath = resolve(here, "../../../../public/push-sw.js");
  const source = readFileSync(swPath, "utf8");

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
    Object,
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
    _clientUrls: Record<string, string>;
    _effectiveUrl: (client: { id: string; url: string }) => string;
    isSafeRelativeUrl: (value: unknown) => boolean;
    resolveIconUrl: (icon: unknown) => string | null;
    resolvePushContent: (
      data: Record<string, unknown>,
      fallback: { title: string; body: string; url: string },
    ) => { title: string; body: string; icon: string; url: string };
  };
}

describe("push-sw getNotificationConfig", () => {
  const sw = loadServiceWorkerHelpers();

  it("returns copilot session_completed config", () => {
    const config = sw.getNotificationConfig({
      type: "copilot_completion",
      event: "session_completed",
    });
    expect(config.title).toBe("AutoGPT");
    expect(config.body).toBe("Task completed");
    expect(config.url).toBe("/copilot");
  });

  it("returns copilot session_failed config", () => {
    const config = sw.getNotificationConfig({
      type: "copilot_completion",
      event: "session_failed",
    });
    expect(config.title).toBe("AutoGPT");
    expect(config.body).toBe("Task failed");
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

  it("returns true when on a different session of the same feature page — sidebar shows the green check, OS popup would be noise", () => {
    const result = sw.isClientViewingTarget(
      client({ url: "https://example.com/copilot?sessionId=A" }),
      "/copilot?sessionId=B",
    );
    expect(result).toBe(true);
  });

  it("ignores query params on either side", () => {
    const result = sw.isClientViewingTarget(
      client({ url: "https://example.com/copilot?utm_source=email" }),
      "/copilot",
    );
    expect(result).toBe(true);
  });
});

describe("push-sw _effectiveUrl", () => {
  const sw = loadServiceWorkerHelpers();

  it("falls back to client.url when no cached URL is set", () => {
    const result = sw._effectiveUrl({
      id: "no-cache",
      url: "https://example.com/copilot",
    });
    expect(result).toBe("https://example.com/copilot");
  });

  it("returns the cached URL when present (resolves against SW origin)", () => {
    sw._clientUrls["client-1"] = "/library";
    const result = sw._effectiveUrl({
      id: "client-1",
      url: "https://example.com/copilot?sessionId=stale",
    });
    expect(result).toBe("https://example.com/library");
  });

  it("preserves query string from the cached URL", () => {
    sw._clientUrls["client-2"] = "/copilot?sessionId=fresh";
    const result = sw._effectiveUrl({
      id: "client-2",
      url: "https://example.com/copilot?sessionId=stale",
    });
    expect(result).toBe("https://example.com/copilot?sessionId=fresh");
  });
});

describe("push-sw notifications toggle", () => {
  const sw = loadServiceWorkerHelpers() as ReturnType<
    typeof loadServiceWorkerHelpers
  > & { _notificationsEnabled: boolean };

  it("defaults to enabled so a freshly-restarted SW still delivers", () => {
    expect(sw._notificationsEnabled).toBe(true);
  });
});

describe("push-sw resolveIconUrl", () => {
  const sw = loadServiceWorkerHelpers();

  it("resolves a site-relative avatar against the SW origin", () => {
    expect(sw.resolveIconUrl("/avatars/maria.webp")).toBe(
      "https://example.com/avatars/maria.webp",
    );
  });

  it("accepts an absolute https avatar URL", () => {
    expect(sw.resolveIconUrl("https://cdn.example.com/a/m.png")).toBe(
      "https://cdn.example.com/a/m.png",
    );
  });

  it("accepts an https avatar URL on any host — the backend is trusted for icon hosts", () => {
    expect(sw.resolveIconUrl("https://evil.example.com/m.png")).toBe(
      "https://evil.example.com/m.png",
    );
  });

  it.each([
    ["missing", undefined],
    ["null", null],
    ["empty", ""],
    ["non-string", 42],
    ["http URL", "http://cdn.example.com/a/m.png"],
    ["data URL", "data:image/png;base64,abc"],
    ["javascript URL", "javascript:alert(1)"],
    ["protocol-relative URL", "//evil.example.com/m.png"],
  ])("returns null for %s", (_label, icon) => {
    expect(sw.resolveIconUrl(icon)).toBeNull();
  });
});

describe("push-sw isSafeRelativeUrl", () => {
  const sw = loadServiceWorkerHelpers();

  it("accepts a same-origin relative path with query", () => {
    expect(sw.isSafeRelativeUrl("/copilot?sessionId=sess-1")).toBe(true);
  });

  it.each([
    ["missing", undefined],
    ["absolute https URL", "https://example.com/copilot"],
    ["protocol-relative URL", "//example.com/copilot"],
    ["bare path", "copilot"],
    ["over-long path", `/${"a".repeat(2048)}`],
  ])("rejects %s", (_label, url) => {
    expect(sw.isSafeRelativeUrl(url)).toBe(false);
  });
});

describe("push-sw resolvePushContent", () => {
  const sw = loadServiceWorkerHelpers();
  const fallback = {
    title: "AutoGPT",
    body: "Task completed",
    url: "/copilot",
  };

  it("prefers server-provided title, body, icon and url", () => {
    const content = sw.resolvePushContent(
      {
        type: "copilot_completion",
        event: "session_completed",
        session_id: "sess-1",
        title: "Maria",
        body: "Finished: Q3 revenue report",
        icon: "/avatars/maria.webp",
        url: "/copilot?sessionId=sess-1",
      },
      fallback,
    );
    expect(content).toEqual({
      title: "Maria",
      body: "Finished: Q3 revenue report",
      icon: "https://example.com/avatars/maria.webp",
      url: "/copilot?sessionId=sess-1",
    });
  });

  it("falls back per-field when the payload carries no rich content", () => {
    const content = sw.resolvePushContent(
      {
        type: "copilot_completion",
        event: "session_completed",
        session_id: "sess-1",
      },
      fallback,
    );
    expect(content).toEqual({
      title: "AutoGPT",
      body: "Task completed",
      icon: "/notification-icon-192.png",
      url: "/copilot?sessionId=sess-1",
    });
  });

  it("builds the session url from session_id when the server url is unsafe", () => {
    const content = sw.resolvePushContent(
      {
        type: "copilot_completion",
        event: "session_completed",
        session_id: "sess-1",
        url: "https://evil.example.com/phish",
      },
      fallback,
    );
    expect(content.url).toBe("/copilot?sessionId=sess-1");
  });

  it("uses the default icon when the server icon is unsafe", () => {
    const content = sw.resolvePushContent(
      {
        type: "copilot_completion",
        event: "session_completed",
        title: "Maria",
        body: "Finished: Q3 revenue report",
        icon: "javascript:alert(1)",
      },
      fallback,
    );
    expect(content.title).toBe("Maria");
    expect(content.icon).toBe("/notification-icon-192.png");
  });

  it("renders rich content for unknown notification types too", () => {
    const content = sw.resolvePushContent(
      {
        type: "future_type",
        event: "something",
        title: "Someone",
        body: "Did something",
      },
      { title: "AutoGPT Notification", body: "something", url: "/" },
    );
    expect(content.title).toBe("Someone");
    expect(content.body).toBe("Did something");
    expect(content.url).toBe("/");
  });
});
