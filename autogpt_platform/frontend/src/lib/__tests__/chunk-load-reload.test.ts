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

const reload = vi.fn();

beforeEach(() => {
  window.sessionStorage.clear();
  capturePostHogEvent.mockClear();
  reload.mockClear();
  vi.spyOn(window, "location", "get").mockReturnValue({
    ...window.location,
    pathname: "/home",
    reload,
  });
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
    expect(reloadForChunkLoadError(chunkLoadError(), "error_boundary")).toBe(
      true,
    );
    expect(reload).toHaveBeenCalledTimes(1);
    expect(capturePostHogEvent).toHaveBeenCalledWith("chunk_reload_triggered", {
      surface: "error_boundary",
      kind: "chunk_load",
      path: "/home",
    });
  });

  it("does not reload again inside the cooldown, so it cannot loop", () => {
    vi.useFakeTimers();

    reloadForChunkLoadError(chunkLoadError(), "error_boundary");
    vi.advanceTimersByTime(30_000);
    expect(reloadForChunkLoadError(chunkLoadError(), "error_boundary")).toBe(
      false,
    );
    expect(reload).toHaveBeenCalledTimes(1);

    vi.advanceTimersByTime(31_000);
    expect(reloadForChunkLoadError(chunkLoadError(), "error_boundary")).toBe(
      true,
    );
    expect(reload).toHaveBeenCalledTimes(2);
  });

  it("does not reload when the guard cannot be stored", () => {
    vi.spyOn(window, "sessionStorage", "get").mockReturnValue({
      getItem: () => null,
      setItem: () => {
        throw new DOMException("blocked", "SecurityError");
      },
    } as unknown as Storage);

    expect(reloadForChunkLoadError(chunkLoadError(), "error_boundary")).toBe(
      false,
    );
    expect(reload).not.toHaveBeenCalled();
    expect(capturePostHogEvent).not.toHaveBeenCalled();
  });

  it("leaves other errors to the error screen", () => {
    expect(reloadForChunkLoadError(new Error("boom"), "error_boundary")).toBe(
      false,
    );
    expect(reload).not.toHaveBeenCalled();
  });
});

// The old tab's webpack runtime is asked for a module the new page names and
// its build never had, so it reads `.call` off undefined.
const WEBPACK_RUNTIME =
  "https://platform.agpt.co/_next/static/chunks/webpack-8e5c5f3b1a7d9c2e.js";
const OUR_CODE =
  "https://platform.agpt.co/_next/static/chunks/app/(platform)/home/page-1f2e3d4c5b6a7980.js";

const MISSING_MODULE_ERRORS = {
  Chrome: {
    message: "Cannot read properties of undefined (reading 'call')",
    frame: (name: string, url: string) => `    at ${name} (${url}:1:143)`,
  },
  Safari: {
    message: "undefined is not an object (evaluating 'e[d].call')",
    frame: (name: string, url: string) => `${name}@${url}:1:143`,
  },
  Firefox: {
    message: `can't access property "call", e[d] is undefined`,
    frame: (name: string, url: string) => `${name}@${url}:1:143`,
  },
};
type Browser = keyof typeof MISSING_MODULE_ERRORS;

function missingModuleError(
  browser: Browser,
  frames: [name: string, url: string][],
) {
  const { message, frame } = MISSING_MODULE_ERRORS[browser];
  const error = new TypeError(message);
  const lines = frames.map(([name, url]) => frame(name, url));
  error.stack = (
    browser === "Chrome" ? [`TypeError: ${message}`, ...lines] : lines
  ).join("\n");
  return error;
}

describe.each(Object.keys(MISSING_MODULE_ERRORS) as Browser[])(
  "a module missing from the old build (%s)",
  (browser) => {
    it("reloads once when webpack's runtime is where it failed", () => {
      const error = missingModuleError(browser, [
        ["r", WEBPACK_RUNTIME],
        ["requireModule", OUR_CODE],
      ]);

      expect(reloadForChunkLoadError(error, "error_boundary")).toBe(true);
      expect(reload).toHaveBeenCalledTimes(1);
      expect(capturePostHogEvent).toHaveBeenCalledWith(
        "chunk_reload_triggered",
        { surface: "error_boundary", kind: "missing_module", path: "/home" },
      );
    });

    it("leaves the same error from our own code to the error screen", () => {
      const error = missingModuleError(browser, [
        ["onClick", OUR_CODE],
        ["r", WEBPACK_RUNTIME],
      ]);

      expect(reloadForChunkLoadError(error, "error_boundary")).toBe(false);
      expect(reload).not.toHaveBeenCalled();
    });
  },
);

describe("reloadOnUnhandledChunkLoadError", () => {
  it("records a reload for a chunk a click handler failed to import", () => {
    const stopListening = reloadOnUnhandledChunkLoadError();
    const event = new Event("unhandledrejection") as PromiseRejectionEvent;
    Object.defineProperty(event, "reason", { value: chunkLoadError() });
    window.dispatchEvent(event);
    stopListening();

    expect(reload).toHaveBeenCalledTimes(1);
    expect(capturePostHogEvent).toHaveBeenCalledWith("chunk_reload_triggered", {
      surface: "unhandled_rejection",
      kind: "chunk_load",
      path: "/home",
    });
  });
});
