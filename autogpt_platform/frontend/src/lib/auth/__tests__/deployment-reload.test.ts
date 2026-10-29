import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const toast = vi.hoisted(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", () => ({ toast }));

const nativeSessionStorage = Object.getOwnPropertyDescriptor(
  window,
  "sessionStorage",
);

// Each call is a fresh page load: module state resets, sessionStorage stays.
async function loadPageAndHitDeploySkew(reload: () => void) {
  vi.resetModules();
  const { reloadForNewDeployment } = await import("../deployment-reload");
  return reloadForNewDeployment(reload);
}

function blockSessionStorage() {
  Object.defineProperty(window, "sessionStorage", {
    configurable: true,
    get() {
      throw new DOMException("The operation is insecure.", "SecurityError");
    },
  });
}

beforeEach(() => {
  window.sessionStorage.clear();
});

afterEach(() => {
  if (nativeSessionStorage) {
    Object.defineProperty(window, "sessionStorage", nativeSessionStorage);
  }
  vi.useRealTimers();
  vi.clearAllMocks();
});

describe("reloading onto a new deployment", () => {
  it("reloads once, then asks the user to refresh instead of looping", async () => {
    const reload = vi.fn();

    expect(await loadPageAndHitDeploySkew(reload)).toBe(true);
    expect(await loadPageAndHitDeploySkew(reload)).toBe(false);
    expect(await loadPageAndHitDeploySkew(reload)).toBe(false);

    expect(reload).toHaveBeenCalledTimes(1);
    expect(toast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "A new version of AutoGPT is available",
      }),
    );
  });

  it("reloads again for a later deployment", async () => {
    vi.useFakeTimers();
    const reload = vi.fn();

    await loadPageAndHitDeploySkew(reload);
    vi.advanceTimersByTime(61_000);
    await loadPageAndHitDeploySkew(reload);

    expect(reload).toHaveBeenCalledTimes(2);
  });

  it("never reloads again once a skew has come back on the reloaded page", async () => {
    vi.useFakeTimers();
    const reload = vi.fn();

    await loadPageAndHitDeploySkew(reload);
    // The reloaded page is rejected too, so reloading can't fix it...
    expect(await loadPageAndHitDeploySkew(reload)).toBe(false);
    // ...and a retry that comes in after the cooldown mustn't try again.
    vi.advanceTimersByTime(61_000);
    expect(await loadPageAndHitDeploySkew(reload)).toBe(false);

    expect(reload).toHaveBeenCalledTimes(1);
    expect(toast).toHaveBeenCalled();
  });

  it("never reloads when sessionStorage is blocked, since the guard can't survive one", async () => {
    blockSessionStorage();
    const reload = vi.fn();

    for (let load = 0; load < 3; load++) {
      expect(await loadPageAndHitDeploySkew(reload)).toBe(false);
    }

    expect(reload).not.toHaveBeenCalled();
    expect(toast).toHaveBeenCalled();
  });
});
