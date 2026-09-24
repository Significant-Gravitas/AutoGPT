"use client";

import { useStoreCategories } from "@/hooks/useStoreCategories";
import { getCategoryHex } from "@/components/molecules/ExpertAvatar/colors";
import { cn } from "@/lib/utils";

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
      aria-label="Browse by category"
    >
      <span className="text-sm font-medium text-zinc-500">
        Browse by category
      </span>
      {/* One flex item, so the whole list drops below the label before it
          breaks within itself. */}
      <div className="flex flex-wrap gap-2.5">
        <CategoryChip
          label="All"
          isSelected={selected === null}
          onClick={() => onSelect(null)}
        />
        {categories.map((category) => (
          <CategoryChip
            key={category.value}
            label={category.label}
            color={getCategoryHex(category.value)}
            title={category.description}
            isSelected={selected === category.value}
            onClick={() => handleClick(category.value)}
          />
        ))}
      </div>
    </div>
  );
}

interface ChipProps {
  label: string;
  title?: string;
  color?: string;
  isSelected: boolean;
  onClick: () => void;
}

function CategoryChip({ label, title, color, isSelected, onClick }: ChipProps) {
  return (
    <button
      type="button"
      title={title}
      aria-pressed={isSelected}
      onClick={onClick}
      className={cn(
        "inline-flex h-9 items-center gap-2 rounded-full border px-4 text-sm font-medium transition-all duration-200",
        isSelected
          ? "border-zinc-900 bg-zinc-900 text-white shadow-[0_1px_2px_rgba(16,24,40,0.1)]"
          : "border-zinc-200 bg-white text-zinc-600 shadow-[0_1px_2px_rgba(16,24,40,0.04)] hover:-translate-y-px hover:border-zinc-300 hover:text-zinc-900",
      )}
    >
      {color && (
        <span
          aria-hidden="true"
          className="size-2 shrink-0 rounded-full ring-1 ring-black/10"
          style={{ backgroundColor: color }}
        />
      )}
      {label}
    </button>
  );
}
