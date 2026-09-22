"use client";

import { useStoreCategories } from "@/hooks/useStoreCategories";
import { cn } from "@/lib/utils";
import { CategoryChip, type ChipSize } from "../CategoryChip/CategoryChip";
import { getCategoryAccent } from "../ExpertsSection/helpers";

interface Props {
  selected: string | null;
  onSelect: (topic: string | null) => void;
  size?: ChipSize;
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
      <CategoryChip
        label="All"
        size={size}
        isSelected={selected === null}
        onClick={() => onSelect(null)}
      />
      {categories.map((category) => {
        const { accent, icon } = getCategoryAccent(category.value);
        return (
          <CategoryChip
            key={category.value}
            label={category.label}
            title={category.description}
            icon={icon}
            accent={accent}
            size={size}
            isSelected={selected === category.value}
            onClick={() =>
              onSelect(selected === category.value ? null : category.value)
            }
          />
        );
      })}
    </div>
  );
}
