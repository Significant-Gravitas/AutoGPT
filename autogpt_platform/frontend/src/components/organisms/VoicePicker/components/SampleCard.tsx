import type { VoiceSample } from "@/app/api/__generated__/models/voiceSample";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";

interface Props {
  sample: VoiceSample;
  isSelected: boolean;
  onSelect: () => void;
}

export function SampleCard({ sample, isSelected, onSelect }: Props) {
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={isSelected}
      className={cn(
        "w-full rounded-2xl border p-5 text-left transition-colors",
        isSelected
          ? "border-purple-300 bg-purple-50/40 ring-2 ring-purple-200"
          : "border-zinc-200 bg-white hover:border-zinc-300",
      )}
    >
      <div className="mb-2 flex items-center justify-between gap-3">
        <span className="text-xs font-medium uppercase tracking-[0.12em] text-purple-600">
          {sample.label}
        </span>
        {isSelected ? (
          <Icon
            icon={CheckmarkCircle02Icon}
            size={18}
            className="shrink-0 text-purple-600"
          />
        ) : null}
      </div>
      <p className="whitespace-pre-line text-[15px] leading-relaxed text-zinc-600">
        {sample.text}
      </p>
    </button>
  );
}
