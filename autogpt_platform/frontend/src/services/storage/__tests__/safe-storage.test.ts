import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@sentry/nextjs", () => ({
  captureException: vi.fn(),
}));

vi.mock("@/services/environment", () => ({
  environment: {
    isServerSide: vi.fn(() => false),
  },
}));

import * as Sentry from "@sentry/nextjs";
import { environment } from "@/services/environment";
import { createSafeStorage } from "../safe-storage";

type TestKey = "a" | "b";

/** Replaces `window.localStorage` for the duration of `run`. Pass a getter that
 *  throws to stand in for a browser that refuses storage outright. */
function withLocalStorage(value: unknown | (() => never), run: () => void) {
  const original = Object.getOwnPropertyDescriptor(window, "localStorage");
  Object.defineProperty(
    window,
    "localStorage",
    typeof value === "function"
      ? { get: value as () => never, configurable: true }
      : { value, configurable: true },
  );
  try {
    run();
  } finally {
    if (original) Object.defineProperty(window, "localStorage", original);
  }
}

describe("createSafeStorage", () => {
  beforeEach(() => {
    window.localStorage.clear();
    window.sessionStorage.clear();
    vi.mocked(Sentry.captureException).mockClear();
    vi.mocked(environment.isServerSide).mockReturnValue(false);
    // The in-memory fallback is module-scoped by design, so drop the keys
    // this file uses between tests. `clean` reaches it as well as the real
    // storage.
    const keys: TestKey[] = ["a", "b"];
    for (const area of ["local", "session"] as const) {
      const storage = createSafeStorage<TestKey>(area);
      keys.forEach((key) => storage.clean(key));
    }
    vi.mocked(Sentry.captureException).mockClear();
  });

  describe("server-side rendering", () => {
    // The whole point of the fix: rendering a page on the server used to
    // report "Local storage is not available" to Sentry on every request.
    it("reads null and reports nothing", () => {
      vi.mocked(environment.isServerSide).mockReturnValue(true);
      const storage = createSafeStorage<TestKey>("local");

      expect(storage.get("a")).toBeNull();
      expect(Sentry.captureException).not.toHaveBeenCalled();
    });

    it("drops writes and reports nothing", () => {
      vi.mocked(environment.isServerSide).mockReturnValue(true);
      const storage = createSafeStorage<TestKey>("local");

      storage.set("a", "1");
      storage.clean("b");

      expect(Sentry.captureException).not.toHaveBeenCalled();
      expect(storage.get("a")).toBeNull();
    });

    it("does not carry a server-side write into the next read", () => {
      // One node process serves every visitor. If the server kept writes in a
      // module-scoped map, visitor B would be served visitor A's value.
      vi.mocked(environment.isServerSide).mockReturnValue(true);
      const storage = createSafeStorage<TestKey>("local");
      storage.set("a", "visitor-a");

      expect(createSafeStorage<TestKey>("local").get("a")).toBeNull();
    });
  });

  describe("storage blocked by the browser", () => {
    it("falls back to memory when reading the storage object throws", () => {
      const storage = createSafeStorage<TestKey>("local");

      withLocalStorage(
        () => {
          throw new Error("The operation is insecure.");
        },
        () => {
          storage.set("a", "kept");
          expect(storage.get("a")).toBe("kept");
          storage.clean("a");
          expect(storage.get("a")).toBeNull();
        },
      );

      expect(Sentry.captureException).not.toHaveBeenCalled();
    });

    it("falls back to memory when the storage object is null", () => {
      const storage = createSafeStorage<TestKey>("local");

      withLocalStorage(null, () => {
        storage.set("a", "kept");
        expect(storage.get("a")).toBe("kept");
      });

      expect(Sentry.captureException).not.toHaveBeenCalled();
    });

    it("keeps the memory fallback out of the real storage once it comes back", () => {
      const storage = createSafeStorage<TestKey>("local");

      withLocalStorage(null, () => storage.set("a", "from-memory"));

      expect(window.localStorage.getItem("a")).toBeNull();
    });
  });

  describe("working storage", () => {
    it("round-trips through the real storage", () => {
      const storage = createSafeStorage<TestKey>("local");

      storage.set("a", "1");
      expect(window.localStorage.getItem("a")).toBe("1");
      expect(storage.get("a")).toBe("1");

      storage.clean("a");
      expect(storage.get("a")).toBeNull();
    });

    it("keeps the two areas apart", () => {
      createSafeStorage<TestKey>("local").set("a", "local-value");
      createSafeStorage<TestKey>("session").set("a", "session-value");

      expect(window.localStorage.getItem("a")).toBe("local-value");
      expect(window.sessionStorage.getItem("a")).toBe("session-value");
    });

    it("reports unexpected write failures and still reads the value back", () => {
      const error = new Error("QuotaExceededError");
      const storage = createSafeStorage<TestKey>("local");
      const setItem = vi
        .spyOn(window.localStorage, "setItem")
        .mockImplementation(() => {
          throw error;
        });

      try {
        storage.set("a", "too-big");
        expect(Sentry.captureException).toHaveBeenCalledWith(error);
        expect(storage.get("a")).toBe("too-big");
      } finally {
        setItem.mockRestore();
      }

      storage.clean("a");
      expect(storage.get("a")).toBeNull();
    });

    it("reads back a failed overwrite of a key that already had a value", () => {
      const storage = createSafeStorage<TestKey>("local");
      storage.set("a", "persisted");

      const setItem = vi
        .spyOn(window.localStorage, "setItem")
        .mockImplementation(() => {
          throw new Error("QuotaExceededError");
        });
      try {
        storage.set("a", "replacement");
        // The old native value must not win over the write that just failed.
        expect(storage.get("a")).toBe("replacement");
      } finally {
        setItem.mockRestore();
      }

      storage.clean("a");
      expect(storage.get("a")).toBeNull();
    });

    it("stops the overlay masking the store once a write succeeds again", () => {
      const storage = createSafeStorage<TestKey>("local");
      const setItem = vi
        .spyOn(window.localStorage, "setItem")
        .mockImplementation(() => {
          throw new Error("QuotaExceededError");
        });
      try {
        storage.set("a", "only-in-memory");
      } finally {
        setItem.mockRestore();
      }
      expect(storage.get("a")).toBe("only-in-memory");

      storage.set("a", "persisted");

      expect(storage.get("a")).toBe("persisted");
      expect(window.localStorage.getItem("a")).toBe("persisted");
    });

    it("reports unexpected removal failures", () => {
      const error = new Error("SecurityError");
      const storage = createSafeStorage<TestKey>("local");
      const removeItem = vi
        .spyOn(window.localStorage, "removeItem")
        .mockImplementation(() => {
          throw error;
        });

      try {
        storage.clean("a");
        expect(Sentry.captureException).toHaveBeenCalledWith(error);
      } finally {
        removeItem.mockRestore();
      }
    });
  });
});
