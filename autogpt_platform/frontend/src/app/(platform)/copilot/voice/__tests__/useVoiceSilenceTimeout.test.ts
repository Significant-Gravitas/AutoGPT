import { renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { Key, storage } from "@/services/storage/local-storage";

import { SILENCE_TIMEOUT_MS } from "../useVoiceMode";
import { useVoiceSilenceTimeout } from "../useVoiceSilenceTimeout";

describe("useVoiceSilenceTimeout", () => {
  afterEach(() => storage.clean(Key.COPILOT_VOICE_SILENCE_TIMEOUT));

  it("defaults to eight seconds", () => {
    expect(renderHook(() => useVoiceSilenceTimeout()).result.current).toBe(
      SILENCE_TIMEOUT_MS,
    );
  });

  it("takes a stored override", () => {
    storage.set(Key.COPILOT_VOICE_SILENCE_TIMEOUT, "20000");
    expect(renderHook(() => useVoiceSilenceTimeout()).result.current).toBe(
      20000,
    );
  });

  it("ignores a value that would strand the mic open or shut it instantly", () => {
    for (const stored of ["500", "600000", "nonsense", ""]) {
      storage.set(Key.COPILOT_VOICE_SILENCE_TIMEOUT, stored);
      expect(renderHook(() => useVoiceSilenceTimeout()).result.current).toBe(
        SILENCE_TIMEOUT_MS,
      );
    }
  });
});
