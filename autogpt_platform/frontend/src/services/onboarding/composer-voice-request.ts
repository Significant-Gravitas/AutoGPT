import { create } from "zustand";

// Otto's first-Home ask can start the composer's recording; the composer
// listens for the count to move (useOnboardingMicGlow).
export const useComposerVoiceRequest = create<{ count: number }>(() => ({
  count: 0,
}));

export function requestComposerVoice() {
  useComposerVoiceRequest.setState((state) => ({ count: state.count + 1 }));
}
