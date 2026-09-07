"use client";

import { useStoreCategories } from "@/hooks/useStoreCategories";
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
      className="mb-8 flex flex-wrap items-center gap-2.5"
      role="group"
      aria-label="Browse by category"
    >
      <span className="mr-1 text-sm font-medium text-zinc-500">
        Browse by category
      </span>
      <CategoryChip
        label="All"
        isSelected={selected === null}
        onClick={() => onSelect(null)}
      />
      {categories.map((category) => (
        <CategoryChip
          key={category.value}
          label={category.label}
          title={category.description}
          isSelected={selected === category.value}
          onClick={() => handleClick(category.value)}
        />
      ))}
    </div>
  );
}

interface ChipProps {
  label: string;
  title?: string;
  isSelected: boolean;
  onClick: () => void;
}

function CategoryChip({ label, title, isSelected, onClick }: ChipProps) {
  return (
    <button
      type="button"
      title={title}
      aria-pressed={isSelected}
      onClick={onClick}
      className={cn(
        "inline-flex h-9 items-center rounded-full border px-4 text-sm font-medium transition-all duration-200",
        isSelected
          ? "border-zinc-900 bg-zinc-900 text-white shadow-[0_1px_2px_rgba(16,24,40,0.1)]"
          : "border-zinc-200 bg-white text-zinc-600 shadow-[0_1px_2px_rgba(16,24,40,0.04)] hover:-translate-y-px hover:border-zinc-300 hover:text-zinc-900",
      )}
    >
      {label}
    </button>
  );
}
