"use client";

import type { VoiceSample } from "@/app/api/__generated__/models/voiceSample";
import { Button } from "@/components/atoms/Button/Button";
import { useId } from "react";
import { SampleCard } from "./components/SampleCard";
import type { VoicePickResult } from "./helpers";
import { selectableCardClassName } from "./styles";
import { useVoicePicker } from "./useVoicePicker";

const MAX_CUSTOM_VOICE_SAMPLE_CHARACTERS = 2_000;
const CUSTOM_VOICE_TEXTAREA_ROWS = 3;

interface Props {
  name?: string;
  samples: VoiceSample[];
  onPick: (result: VoicePickResult) => void;
  onSkip: () => void;
  isSubmitting?: boolean;
}

export function VoicePicker({
  name,
  samples,
  onPick,
  onSkip,
  isSubmitting = false,
}: Props) {
  const choiceGroupName = useId();
  const customTextareaId = useId();
  const customCharacterCountId = `${customTextareaId}-character-count`;
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

        <div className={selectableCardClassName(selected === "custom")}>
          <label
            htmlFor={`${customTextareaId}-choice`}
            className="mb-2 block cursor-pointer text-xs font-medium uppercase tracking-[0.12em] text-accent"
          >
            <input
              id={`${customTextareaId}-choice`}
              type="radio"
              name={choiceGroupName}
              value="custom"
              checked={selected === "custom"}
              onChange={focusCustom}
              className="sr-only"
            />
            Paste your own
          </label>
          <textarea
            id={customTextareaId}
            value={customText}
            onFocus={focusCustom}
            onChange={(event) => changeCustom(event.target.value)}
            rows={CUSTOM_VOICE_TEXTAREA_ROWS}
            maxLength={MAX_CUSTOM_VOICE_SAMPLE_CHARACTERS}
            aria-describedby={customCharacterCountId}
            placeholder="Paste a few sentences written the way you'd like this expert to sound."
            className="w-full resize-none rounded-xl border border-input bg-background px-4 py-2.5 text-sm leading-relaxed text-foreground placeholder:text-muted-foreground focus:border-ring focus:outline-none focus:ring-1 focus:ring-ring"
          />
          <p
            id={customCharacterCountId}
            className="mt-1.5 text-right text-xs text-muted-foreground"
          >
            {customText.length.toLocaleString()} /{" "}
            {MAX_CUSTOM_VOICE_SAMPLE_CHARACTERS.toLocaleString()} characters
          </p>
        </div>
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
