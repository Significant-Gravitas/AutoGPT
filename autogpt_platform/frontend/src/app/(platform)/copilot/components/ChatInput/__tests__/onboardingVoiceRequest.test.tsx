import { requestComposerVoice } from "@/services/onboarding/composer-voice-request";
import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useOnboardingMicGlow } from "../useOnboardingMicGlow";

describe("Otto's first-Home ask starting the composer's recording", () => {
  it.each([
    { canRecord: true, calls: 1 },
    { canRecord: false, calls: 0 },
  ])(
    "records on request only when the mic is free (canRecord $canRecord)",
    ({ canRecord, calls }) => {
      const startRecording = vi.fn();
      renderHook(() =>
        useOnboardingMicGlow({
          isTranscribing: false,
          canRecord,
          startRecording,
        }),
      );
      expect(startRecording).not.toHaveBeenCalled();

      act(() => requestComposerVoice());

      expect(startRecording).toHaveBeenCalledTimes(calls);
    },
  );
});
