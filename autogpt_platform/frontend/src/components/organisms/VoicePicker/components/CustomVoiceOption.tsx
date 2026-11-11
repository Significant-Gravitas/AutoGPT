import { cn } from "@/lib/utils";
import { selectableCardClassName, type SelectableCardColors } from "../helpers";

const MAX_CUSTOM_VOICE_SAMPLE_CHARACTERS = 2_000;
const CUSTOM_VOICE_TEXTAREA_ROWS = 3;

interface Props {
  choiceGroupName: string;
  textareaId: string;
  customText: string;
  isSelected: boolean;
  labelClassName?: string;
  colors?: SelectableCardColors;
  compact?: boolean;
  onFocus: () => void;
  onChange: (value: string) => void;
}

export function CustomVoiceOption({
  choiceGroupName,
  textareaId,
  customText,
  isSelected,
  labelClassName,
  colors,
  compact = false,
  onFocus,
  onChange,
}: Props) {
  const characterCountId = `${textareaId}-character-count`;

  return (
    <div
      data-slot="custom-voice-option"
      className={selectableCardClassName(isSelected, false, colors, compact)}
    >
      <label
        htmlFor={`${textareaId}-choice`}
        data-slot="custom-voice-option-label"
        className={cn(
          "mb-2 block cursor-pointer text-caption font-semibold tracking-caps uppercase",
          labelClassName ?? "text-brand",
        )}
      >
        <input
          id={`${textareaId}-choice`}
          type="radio"
          name={choiceGroupName}
          value="custom"
          checked={isSelected}
          onChange={onFocus}
          data-slot="custom-voice-option-radio"
          className="sr-only"
        />
        Paste your own
      </label>
      <label
        htmlFor={textareaId}
        data-slot="custom-voice-option-textarea-label"
        className="sr-only"
      >
        Custom voice sample
      </label>
      <textarea
        id={textareaId}
        value={customText}
        onFocus={onFocus}
        onChange={(event) => onChange(event.target.value)}
        rows={compact ? 2 : CUSTOM_VOICE_TEXTAREA_ROWS}
        maxLength={MAX_CUSTOM_VOICE_SAMPLE_CHARACTERS}
        aria-describedby={characterCountId}
        placeholder="Paste a few sentences written the way you'd like this expert to sound."
        data-slot="custom-voice-option-textarea"
        className={cn(
          "w-full resize-none border bg-canvas text-body leading-relaxed text-foreground-base placeholder:text-foreground-tertiary focus:ring-1 focus:ring-focus focus:outline-none",
          compact ? "rounded-md px-3 py-2" : "rounded-xl px-4 py-2.5",
          isSelected
            ? "border-focus"
            : "border-border-strong focus:border-focus",
        )}
      />
      <p
        id={characterCountId}
        data-slot="custom-voice-option-count"
        className="mt-1.5 text-right text-caption text-foreground-tertiary"
      >
        {customText.length.toLocaleString()} /{" "}
        {MAX_CUSTOM_VOICE_SAMPLE_CHARACTERS.toLocaleString()} characters
      </p>
    </div>
  );
}
