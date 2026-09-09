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
  compact?: boolean;
  onSelect: () => void;
};

export function SampleCard({
  sample,
  choice,
  choiceGroupName,
  isSelected,
  labelClassName,
  colors,
  compact = false,
  onSelect,
}: Props) {
  return (
    <label
      className={cn(
        "block w-full text-left",
        selectableCardClassName(isSelected, true, colors, compact),
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
      <div
        className={cn(
          "flex items-center justify-between gap-3",
          compact ? "mb-1" : "mb-2",
        )}
      >
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
      <p
        className={cn(
          "whitespace-pre-line text-muted-foreground",
          compact
            ? "line-clamp-3 text-sm leading-normal"
            : "text-[15px] leading-relaxed",
        )}
      >
        {sample.text}
      </p>
    </label>
  );
}
