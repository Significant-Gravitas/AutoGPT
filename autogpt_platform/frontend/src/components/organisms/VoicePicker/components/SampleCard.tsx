import type { VoiceSample } from "@/app/api/__generated__/models/voiceSample";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";
import { selectableCardClassName, type SelectableCardColors } from "../helpers";

interface Props {
  sample: VoiceSample;
  choice: "a" | "b";
  choiceGroupName: string;
  isSelected: boolean;
  labelClassName?: string;
  colors?: SelectableCardColors;
  compact?: boolean;
  onSelect: () => void;
}

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
      data-slot="sample-card"
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
        data-slot="sample-card-radio"
        className="sr-only"
      />
      <div
        data-slot="sample-card-header"
        className={cn(
          "flex items-center justify-between gap-3",
          compact ? "mb-1" : "mb-2",
        )}
      >
        <span
          data-slot="sample-card-label"
          className={cn(
            "text-xs font-semibold uppercase tracking-caps",
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
        data-slot="sample-card-text"
        className={cn(
          "whitespace-pre-line text-muted-foreground",
          compact
            ? "line-clamp-3 text-sm leading-normal"
            : "text-body-md leading-relaxed",
        )}
      >
        {sample.text}
      </p>
    </label>
  );
}
