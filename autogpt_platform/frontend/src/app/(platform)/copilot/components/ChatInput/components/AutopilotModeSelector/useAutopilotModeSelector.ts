import { useState } from "react";
import {
  type AutopilotMode,
  DEFAULT_AUTOPILOT_MODE,
  useAutopilotModeChoice,
  useAutopilotModeStore,
} from "../../../../autopilotModeStore";
import { isAutopilotMode } from "./helpers";

interface Args {
  sessionId: string | null;
  persistedMode: AutopilotMode | null;
}

export function useAutopilotModeSelector({ sessionId, persistedMode }: Args) {
  const choice = useAutopilotModeChoice(sessionId);
  const choose = useAutopilotModeStore((state) => state.choose);
  const [isConfirmOpen, setIsConfirmOpen] = useState(false);
  const mode = choice ?? persistedMode ?? DEFAULT_AUTOPILOT_MODE;

  function selectMode(value: string) {
    if (!isAutopilotMode(value) || value === mode) return;
    if (value === "unsupervised") {
      setIsConfirmOpen(true);
      return;
    }
    choose(sessionId, value);
  }

  function confirmUnsupervised() {
    choose(sessionId, "unsupervised");
    setIsConfirmOpen(false);
  }

  function cancelUnsupervised() {
    setIsConfirmOpen(false);
  }

  return {
    mode,
    isDefault: mode === DEFAULT_AUTOPILOT_MODE,
    selectMode,
    isConfirmOpen,
    confirmUnsupervised,
    cancelUnsupervised,
  };
}
