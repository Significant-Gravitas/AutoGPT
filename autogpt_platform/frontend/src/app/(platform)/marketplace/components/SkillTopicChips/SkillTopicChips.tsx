"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { useStoreCategories } from "@/hooks/useStoreCategories";
import { cn } from "@/lib/utils";
import { getCategoryAccent } from "../ExpertsSection/helpers";

type Size = "small" | "default";

interface Props {
  selected: string | null;
  onSelect: (topic: string | null) => void;
  /** "small" tucks the row into a section header; "default" is a page's own
   *  filter and matches the height of the marketplace category pills. */
  size?: Size;
}

export function SkillTopicChips({ selected, onSelect, size = "small" }: Props) {
  const { categories } = useStoreCategories();

  if (categories.length === 0) return null;

  return (
    <div
      role="group"
      aria-label="Filter skills by topic"
      className={cn(
        "flex flex-wrap",
        size === "small" ? "gap-1 sm:justify-end" : "gap-2",
      )}
    >
      <TopicChip
        label="All"
        size={size}
        isSelected={selected === null}
        onClick={() => onSelect(null)}
      />
      {categories.map((category) => (
        <TopicChip
          key={category.value}
          topic={category.value}
          label={category.label}
          title={category.description}
          size={size}
          isSelected={selected === category.value}
          onClick={() =>
            onSelect(selected === category.value ? null : category.value)
          }
        />
      ))}
    </div>
  );
}

interface ChipProps {
  topic?: string;
  label: string;
  title?: string;
  size: Size;
  isSelected: boolean;
  onClick: () => void;
}

function TopicChip({
  topic,
  label,
  title,
  size,
  isSelected,
  onClick,
}: ChipProps) {
  const { accent, icon } = getCategoryAccent(topic);

  return (
    <button
      type="button"
      title={title}
      aria-pressed={isSelected}
      onClick={onClick}
      className={cn(
        "inline-flex items-center rounded-full font-medium outline-none transition-colors focus-visible:ring-2 focus-visible:ring-violet-600",
        size === "small"
          ? "h-7 gap-1 px-2 text-xs"
          : "h-9 gap-1.5 px-3.5 text-sm",
        isSelected
          ? topic
            ? accent.pill
            : "bg-zinc-900 text-white"
          : "bg-white text-zinc-600 ring-1 ring-inset ring-zinc-200 hover:text-zinc-900 hover:ring-zinc-300",
      )}
    >
      {icon ? (
        <Icon
          icon={icon}
          size={size === "small" ? 12 : 15}
          className={isSelected ? undefined : accent.icon}
          aria-hidden
        />
      ) : null}
      {label}
    </button>
  );
}
