"use client";

import type { VoiceSample } from "@/app/api/__generated__/models/voiceSample";
import { Button } from "@/components/atoms/Button/Button";
import { useId } from "react";
import { CustomVoiceOption } from "./components/CustomVoiceOption";
import { SampleCard } from "./components/SampleCard";
import type { VoicePickResult } from "./helpers";
import { useVoicePicker } from "./useVoicePicker";

type Props = {
  name?: string;
  samples: VoiceSample[];
  onPick: (result: VoicePickResult) => void;
  onSkip: () => void;
  isSubmitting?: boolean;
};

export function VoicePicker({
  name,
  samples,
  onPick,
  onSkip,
  isSubmitting = false,
}: Props) {
  const choiceGroupName = useId();
  const customTextareaId = useId();
  const {
    selected,
    customText,
    selectSample,
    focusCustom,
    changeCustom,
    canSubmit,
    submit,
  } = useVoicePicker({ onPick });

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1.5">
        <h2 className="text-2xl font-semibold tracking-[-0.02em] text-foreground">
          {name ? `How should ${name} write?` : "How should this expert write?"}
        </h2>
        <p className="text-base text-muted-foreground">
          Pick the voice that feels right. You can fine-tune it anytime in the
          Soul editor.
        </p>
      </header>

      <fieldset className="flex flex-col gap-3">
        <legend className="sr-only">Writing voice</legend>
        {samples.slice(0, 2).map((sample, index) => {
          const choice = index === 0 ? "a" : "b";
          return (
            <SampleCard
              key={choice}
              sample={sample}
              choice={choice}
              choiceGroupName={choiceGroupName}
              isSelected={selected === choice}
              onSelect={() => selectSample(choice)}
            />
          );
        })}

        <CustomVoiceOption
          choiceGroupName={choiceGroupName}
          textareaId={customTextareaId}
          customText={customText}
          isSelected={selected === "custom"}
          onFocus={focusCustom}
          onChange={changeCustom}
        />
      </fieldset>

      <footer className="flex items-center justify-between gap-3">
        <Button variant="ghost" onClick={onSkip} disabled={isSubmitting}>
          Skip for now
        </Button>
        <Button
          variant="primary"
          onClick={submit}
          disabled={!canSubmit}
          loading={isSubmitting}
          className="rounded-full"
        >
          Use this voice
        </Button>
      </footer>
    </div>
  );
}
