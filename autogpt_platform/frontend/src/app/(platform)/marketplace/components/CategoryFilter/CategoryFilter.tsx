"use client";

import { useStoreCategories } from "@/hooks/useStoreCategories";
import { CategoryChip } from "../CategoryChip/CategoryChip";
import { getCategoryAccent } from "../ExpertsSection/helpers";

interface Props {
  selected: string | null;
  onSelect: (category: string | null) => void;
}

export function CategoryFilter({ selected, onSelect }: Props) {
  const { categories } = useStoreCategories();

  if (categories.length === 0) return null;

  function handleClick(value: string | null) {
    onSelect(value === selected ? null : value);
  }

  return (
    // Labelled because the hero above carries its own chip row of search
    // terms, and the two overlap on names like "Marketing".
    <div
      className="mb-8 flex flex-wrap items-center gap-x-4 gap-y-2.5"
      role="group"
      aria-label="Browse experts by category"
    >
      <span className="text-sm font-medium text-zinc-500">
        Browse experts by category
      </span>
      {/* One flex item, so the whole list drops below the label before it
          breaks within itself. */}
      <div className="flex flex-wrap gap-2">
        <CategoryChip
          label="All"
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
              isSelected={selected === category.value}
              onClick={() => handleClick(category.value)}
            />
          );
        })}
      </div>
    </div>
  );
}
