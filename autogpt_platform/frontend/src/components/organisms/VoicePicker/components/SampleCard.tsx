import type { VoiceSample } from "@/app/api/__generated__/models/voiceSample";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";
import { selectableCardClassName } from "../styles";

interface Props {
  sample: VoiceSample;
  choice: "a" | "b";
  choiceGroupName: string;
  isSelected: boolean;
  onSelect: () => void;
}

export function SampleCard({
  sample,
  choice,
  choiceGroupName,
  isSelected,
  onSelect,
}: Props) {
  return (
    <label
      className={cn(
        "block w-full text-left",
        selectableCardClassName(isSelected, true),
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
        <span className="text-xs font-medium uppercase tracking-[0.12em] text-accent">
          {sample.label}
        </span>
        {isSelected ? (
          <Icon
            icon={CheckmarkCircle02Icon}
            size={18}
            className="shrink-0 text-accent"
          />
        ) : null}
      </div>
      <p className="whitespace-pre-line text-[15px] leading-relaxed text-muted-foreground">
        {sample.text}
      </p>
    </label>
  );
}
