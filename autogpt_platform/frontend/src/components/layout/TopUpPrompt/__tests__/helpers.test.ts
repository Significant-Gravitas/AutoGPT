import { describe, expect, it, beforeEach } from "vitest";
import { storage, Key } from "@/services/storage/local-storage";
import { wasShownToday, markShownToday } from "../helpers";

describe("TopUpPrompt date helpers", () => {
  beforeEach(() => storage.clean(Key.TOP_UP_MODAL_LAST_SHOWN));

  it("returns false when never shown", () => {
    expect(wasShownToday(Key.TOP_UP_MODAL_LAST_SHOWN)).toBe(false);
  });

  it("returns true after marking today", () => {
    markShownToday(Key.TOP_UP_MODAL_LAST_SHOWN);
    expect(wasShownToday(Key.TOP_UP_MODAL_LAST_SHOWN)).toBe(true);
  });

  it("returns false for a stale (yesterday) stamp", () => {
    const yesterday = new Date(Date.now() - 86_400_000).toDateString();
    storage.set(Key.TOP_UP_MODAL_LAST_SHOWN, yesterday);
    expect(wasShownToday(Key.TOP_UP_MODAL_LAST_SHOWN)).toBe(false);
  });
});
