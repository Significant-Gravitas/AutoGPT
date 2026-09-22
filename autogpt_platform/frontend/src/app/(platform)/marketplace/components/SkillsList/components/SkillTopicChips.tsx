"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { useStoreCategories } from "@/hooks/useStoreCategories";
import { cn } from "@/lib/utils";
import { getCategoryAccent } from "../../ExpertsSection/helpers";

interface Props {
  selected: string | null;
  onSelect: (topic: string | null) => void;
}

export function SkillTopicChips({ selected, onSelect }: Props) {
  const { categories } = useStoreCategories();

  if (categories.length === 0) return null;

  return (
    <div
      role="group"
      aria-label="Filter skills by topic"
      className="flex flex-wrap gap-1 sm:justify-end"
    >
      <TopicChip
        label="All"
        isSelected={selected === null}
        onClick={() => onSelect(null)}
      />
      {categories.map((category) => (
        <TopicChip
          key={category.value}
          topic={category.value}
          label={category.label}
          title={category.description}
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
  isSelected: boolean;
  onClick: () => void;
}

function TopicChip({ topic, label, title, isSelected, onClick }: ChipProps) {
  const { accent, icon } = getCategoryAccent(topic);

  return (
    <button
      type="button"
      title={title}
      aria-pressed={isSelected}
      onClick={onClick}
      className={cn(
        "inline-flex h-7 items-center gap-1 rounded-full px-2 text-xs font-medium outline-none transition-colors focus-visible:ring-2 focus-visible:ring-violet-600",
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
          size={12}
          className={isSelected ? undefined : accent.icon}
          aria-hidden
        />
      ) : null}
      {label}
    </button>
  );
}
