import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const capturePostHogEvent = vi.hoisted(() => vi.fn());
vi.mock("@/services/analytics/posthog-capture", () => ({
  capturePostHogEvent,
}));

import {
  isChunkLoadError,
  reloadForChunkLoadError,
  reloadOnUnhandledChunkLoadError,
} from "../chunk-load-reload";

function chunkLoadError() {
  const error = new Error(
    "Loading chunk 34006 failed.\n(error: https://platform.agpt.co/_next/static/chunks/34006.0781329c05fff018.js)",
  );
  error.name = "ChunkLoadError";
  return error;
}

beforeEach(() => {
  window.sessionStorage.clear();
  capturePostHogEvent.mockClear();
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("isChunkLoadError", () => {
  it("matches webpack's JS and CSS chunk failures and Turbopack's", () => {
    expect(isChunkLoadError(chunkLoadError())).toBe(true);
    expect(isChunkLoadError(new Error("Loading CSS chunk 123 failed."))).toBe(
      true,
    );
    expect(
      isChunkLoadError(
        new Error("Failed to load chunk /_next/static/chunks/abc.js"),
      ),
    ).toBe(true);
  });

  it("ignores other errors", () => {
    expect(isChunkLoadError(new Error("Failed to fetch"))).toBe(false);
    expect(isChunkLoadError("Loading chunk 1 failed.")).toBe(false);
    expect(isChunkLoadError(undefined)).toBe(false);
  });
});

describe("reloadForChunkLoadError", () => {
  it("reloads once and records the reload", () => {
    const reload = vi.fn();

    expect(
      reloadForChunkLoadError(chunkLoadError(), "error_boundary", reload),
    ).toBe(true);
    expect(reload).toHaveBeenCalledTimes(1);
    expect(capturePostHogEvent).toHaveBeenCalledWith("chunk_reload_triggered", {
      surface: "error_boundary",
      path: window.location.pathname,
    });
  });

  it("does not reload again inside the cooldown, so it cannot loop", () => {
    vi.useFakeTimers();
    const reload = vi.fn();

    reloadForChunkLoadError(chunkLoadError(), "error_boundary", reload);
    vi.advanceTimersByTime(30_000);
    expect(
      reloadForChunkLoadError(chunkLoadError(), "error_boundary", reload),
    ).toBe(false);
    expect(reload).toHaveBeenCalledTimes(1);

    vi.advanceTimersByTime(31_000);
    expect(
      reloadForChunkLoadError(chunkLoadError(), "error_boundary", reload),
    ).toBe(true);
    expect(reload).toHaveBeenCalledTimes(2);
  });

  it("does not reload when the guard cannot be stored", () => {
    vi.spyOn(window, "sessionStorage", "get").mockReturnValue({
      getItem: () => null,
      setItem: () => {
        throw new DOMException("blocked", "SecurityError");
      },
    } as unknown as Storage);
    const reload = vi.fn();

    expect(
      reloadForChunkLoadError(chunkLoadError(), "error_boundary", reload),
    ).toBe(false);
    expect(reload).not.toHaveBeenCalled();
    expect(capturePostHogEvent).not.toHaveBeenCalled();
  });

  it("leaves other errors to the error screen", () => {
    const reload = vi.fn();

    expect(
      reloadForChunkLoadError(new Error("boom"), "error_boundary", reload),
    ).toBe(false);
    expect(reload).not.toHaveBeenCalled();
  });
});

describe("reloadOnUnhandledChunkLoadError", () => {
  it("records a reload for a chunk a click handler failed to import", () => {
    const stopListening = reloadOnUnhandledChunkLoadError();
    const reload = vi.fn();
    vi.spyOn(window, "location", "get").mockReturnValue({
      ...window.location,
      pathname: "/home",
      reload,
    });

    const event = new Event("unhandledrejection") as PromiseRejectionEvent;
    Object.defineProperty(event, "reason", { value: chunkLoadError() });
    window.dispatchEvent(event);
    stopListening();

    expect(reload).toHaveBeenCalledTimes(1);
    expect(capturePostHogEvent).toHaveBeenCalledWith("chunk_reload_triggered", {
      surface: "unhandled_rejection",
      path: "/home",
    });
  });
});
