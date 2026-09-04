import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  captureFirstLanding,
  getAnonymousID,
  getPostHogDeviceID,
  readFirstLanding,
  resetAnonymousID,
  resetAnonymousIDForTests,
} from "../anonymous-id";

const ANONYMOUS_ID_KEY = "agpt_anonymous_id";
const FIRST_LANDING_KEY = "agpt_first_landing";

beforeEach(() => {
  window.localStorage.clear();
  resetAnonymousIDForTests();
  vi.stubEnv("NEXT_PUBLIC_POSTHOG_KEY", "phc_test");
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("getAnonymousID", () => {
  it("mints one id, persists it, and returns the same id afterwards", () => {
    const first = getAnonymousID();

    expect(first).toBeTruthy();
    expect(window.localStorage.getItem(ANONYMOUS_ID_KEY)).toBe(first);
    expect(getAnonymousID()).toBe(first);

    resetAnonymousIDForTests();
    expect(getAnonymousID()).toBe(first);
  });

  it("adopts an existing PostHog device id so returning visitors keep history", () => {
    window.localStorage.setItem(
      "ph_phc_test_posthog",
      JSON.stringify({ $device_id: "device-123" }),
    );

    expect(getAnonymousID()).toBe("device-123");
    expect(getPostHogDeviceID()).toBe("device-123");
  });

  it("ignores unreadable PostHog persistence", () => {
    window.localStorage.setItem("ph_phc_test_posthog", "not json");

    expect(getPostHogDeviceID()).toBeNull();
    expect(getAnonymousID()).not.toBe("not json");
  });

  it("keeps an in-memory id when storage is blocked", () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });

    const id = getAnonymousID();

    expect(id).toBeTruthy();
    expect(getAnonymousID()).toBe(id);
  });
});

describe("resetAnonymousID", () => {
  it("forgets the id so the next visitor starts as a new person", () => {
    const first = getAnonymousID();

    resetAnonymousID();

    expect(window.localStorage.getItem(ANONYMOUS_ID_KEY)).toBeNull();
    expect(getAnonymousID()).not.toBe(first);
  });
});

describe("first landing", () => {
  it("records the first page once and reads it back", () => {
    window.history.replaceState(
      null,
      "",
      "/pricing?utm_source=x&utm_medium=cpc&utm_campaign=launch",
    );

    captureFirstLanding();
    window.history.replaceState(null, "", "/somewhere-else");
    captureFirstLanding();

    const landing = readFirstLanding();
    expect(landing?.path).toBe(
      "/pricing?utm_source=x&utm_medium=cpc&utm_campaign=launch",
    );
    expect(landing?.utm_source).toBe("x");
    expect(landing?.utm_medium).toBe("cpc");
    expect(landing?.utm_campaign).toBe("launch");
    expect(landing?.referrer).toBeNull();
    expect(landing?.at).toBeTruthy();
  });

  it("returns null when nothing was captured or the record is corrupt", () => {
    expect(readFirstLanding()).toBeNull();

    window.localStorage.setItem(FIRST_LANDING_KEY, "{");
    expect(readFirstLanding()).toBeNull();
  });
});
