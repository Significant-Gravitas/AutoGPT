import type { VoiceSample } from "@/app/api/__generated__/models/voiceSample";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";
import { selectableCardClassName, type SelectableCardColors } from "../styles";

type Props = {
  sample: VoiceSample;
  choice: "a" | "b";
  choiceGroupName: string;
  isSelected: boolean;
  labelClassName?: string;
  colors?: SelectableCardColors;
  onSelect: () => void;
};

export function SampleCard({
  sample,
  choice,
  choiceGroupName,
  isSelected,
  labelClassName,
  colors,
  onSelect,
}: Props) {
  return (
    <label
      className={cn(
        "block w-full text-left",
        selectableCardClassName(isSelected, true, colors),
      )}
    >
      <input
        type="radio"
        name={choiceGroupName}
        value={choice}
        checked={isSelected}
        onChange={onSelect}
        aria-label={sample.label}
        className="sr-only"
      />
      <div className="mb-2 flex items-center justify-between gap-3">
        <span
          className={cn(
            "text-xs font-semibold uppercase tracking-[0.12em]",
            labelClassName ?? "text-accent",
          )}
        >
          {sample.label}
        </span>
        {isSelected ? (
          <Icon
            icon={CheckmarkCircle02Icon}
            size={18}
            className={cn("shrink-0", labelClassName ?? "text-accent")}
          />
        ) : null}
      </div>
      <p className="whitespace-pre-line text-[15px] leading-relaxed text-muted-foreground">
        {sample.text}
      </p>
    </label>
  );
}
