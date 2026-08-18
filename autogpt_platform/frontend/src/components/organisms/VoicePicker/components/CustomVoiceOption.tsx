import { cn } from "@/lib/utils";
import { selectableCardClassName, type SelectableCardColors } from "../styles";

const MAX_CUSTOM_VOICE_SAMPLE_CHARACTERS = 2_000;
const CUSTOM_VOICE_TEXTAREA_ROWS = 3;

type Props = {
  choiceGroupName: string;
  textareaId: string;
  customText: string;
  isSelected: boolean;
  labelClassName?: string;
  colors?: SelectableCardColors;
  onFocus: () => void;
  onChange: (value: string) => void;
};

export function CustomVoiceOption({
  choiceGroupName,
  textareaId,
  customText,
  isSelected,
  labelClassName,
  colors,
  onFocus,
  onChange,
}: Props) {
  const characterCountId = `${textareaId}-character-count`;

  return (
    <div className={selectableCardClassName(isSelected, false, colors)}>
      <label
        htmlFor={`${textareaId}-choice`}
        className={cn(
          "mb-2 block cursor-pointer text-xs font-semibold uppercase tracking-[0.12em]",
          labelClassName ?? "text-accent",
        )}
      >
        <input
          id={`${textareaId}-choice`}
          type="radio"
          name={choiceGroupName}
          value="custom"
          checked={isSelected}
          onChange={onFocus}
          className="sr-only"
        />
        Paste your own
      </label>
      <label htmlFor={textareaId} className="sr-only">
        Custom voice sample
      </label>
      <textarea
        id={textareaId}
        value={customText}
        onFocus={onFocus}
        onChange={(event) => onChange(event.target.value)}
        rows={CUSTOM_VOICE_TEXTAREA_ROWS}
        maxLength={MAX_CUSTOM_VOICE_SAMPLE_CHARACTERS}
        aria-describedby={characterCountId}
        placeholder="Paste a few sentences written the way you'd like this expert to sound."
        className={cn(
          "w-full resize-none rounded-xl border bg-background px-4 py-2.5 text-sm leading-relaxed text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring",
          isSelected ? "border-ring" : "border-input focus:border-ring",
        )}
      />
      <p
        id={characterCountId}
        className="mt-1.5 text-right text-xs text-muted-foreground"
      >
        {customText.length.toLocaleString()} /{" "}
        {MAX_CUSTOM_VOICE_SAMPLE_CHARACTERS.toLocaleString()} characters
      </p>
    </div>
  );
}
