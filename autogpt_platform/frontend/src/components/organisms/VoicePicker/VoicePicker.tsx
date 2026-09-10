"use client";

import type { VoiceSample } from "@/app/api/__generated__/models/voiceSample";
import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";
import { useId } from "react";
import { CustomVoiceOption } from "./components/CustomVoiceOption";
import { SampleCard } from "./components/SampleCard";
import type { VoicePickResult } from "./helpers";
import type { SelectableCardColors } from "./helpers";
import { useVoicePicker } from "./useVoicePicker";

interface Props {
  name?: string;
  samples: VoiceSample[];
  onPick: (result: VoicePickResult) => void;
  onSkip: () => void;
  isSubmitting?: boolean;
  // Set when the surrounding flow already asked the question in its own copy.
  hideHeader?: boolean;
  // Overrides the accent used on option labels (e.g. the expert's colour).
  labelClassName?: string;
  // Overrides the card's selected/hover/focus treatment to match that colour.
  cardColors?: SelectableCardColors;
  // Dense layout for a small dialog: tighter spacing, smaller type.
  compact?: boolean;
}

export function VoicePicker({
  name,
  samples,
  onPick,
  onSkip,
  isSubmitting = false,
  hideHeader = false,
  labelClassName,
  cardColors,
  compact = false,
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
    <div className={cn("flex flex-col", compact ? "gap-4" : "gap-6")}>
      {hideHeader ? null : (
        <header className={cn("flex flex-col", compact ? "gap-1" : "gap-1.5")}>
          <h2
            className={cn(
              "text-foreground",
              compact
                ? "text-base font-medium"
                : "text-2xl font-semibold tracking-[-0.02em]",
            )}
          >
            {name
              ? `How should ${name} write?`
              : "How should this expert write?"}
          </h2>
          <p
            className={cn(
              "text-muted-foreground",
              compact ? "text-sm" : "text-base",
            )}
          >
            Pick the voice that feels right. You can fine-tune it anytime in the
            Soul editor.
          </p>
        </header>
      )}

      <fieldset className={cn("flex flex-col", compact ? "gap-2" : "gap-3")}>
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
              labelClassName={labelClassName}
              colors={cardColors}
              compact={compact}
              onSelect={() => selectSample(choice)}
            />
          );
        })}

        <CustomVoiceOption
          choiceGroupName={choiceGroupName}
          textareaId={customTextareaId}
          customText={customText}
          isSelected={selected === "custom"}
          labelClassName={labelClassName}
          colors={cardColors}
          compact={compact}
          onFocus={focusCustom}
          onChange={changeCustom}
        />
      </fieldset>

      <footer className="flex items-center justify-between gap-3">
        <Button
          variant="ghost"
          size={compact ? "small" : undefined}
          onClick={onSkip}
          disabled={isSubmitting}
        >
          Skip for now
        </Button>
        <Button
          variant="primary"
          size={compact ? "small" : undefined}
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
