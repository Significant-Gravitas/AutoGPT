import { useState } from "react";
import type { VoicePickChoice, VoicePickResult } from "./helpers";

interface Args {
  onPick: (result: VoicePickResult) => void;
}

export function useVoicePicker({ onPick }: Args) {
  const [selected, setSelected] = useState<VoicePickChoice | null>(null);
  const [customText, setCustomText] = useState("");

  function selectSample(choice: "a" | "b") {
    setSelected(choice);
  }

  function focusCustom() {
    setSelected("custom");
  }

  function changeCustom(value: string) {
    setCustomText(value);
    setSelected("custom");
  }

  const canSubmit =
    selected === "custom" ? customText.trim().length > 0 : selected !== null;

  function submit() {
    if (!selected || !canSubmit) return;
    onPick(
      selected === "custom"
        ? { choice: "custom", customText }
        : { choice: selected },
    );
  }

  return {
    selected,
    customText,
    selectSample,
    focusCustom,
    changeCustom,
    canSubmit,
    submit,
  };
}
