import { useBlockMenuStore } from "@/app/(platform)/build/stores/blockMenuStore";
import { useShallow } from "zustand/react/shallow";
import { FilterChip } from "../FilterChip";
import { categories } from "./constants";
import { FilterSheet } from "../FilterSheet/FilterSheet";
import { CategoryKey } from "./types";

export const BlockMenuFilters = () => {
  const {
    filters,
    addFilter,
    removeFilter,
    categoryCounts,
    creators,
    addCreator,
    removeCreator,
  } = useBlockMenuStore(
    useShallow((state) => ({
      filters: state.filters,
      addFilter: state.addFilter,
      removeFilter: state.removeFilter,
      categoryCounts: state.categoryCounts,
      creators: state.creators,
      addCreator: state.addCreator,
      removeCreator: state.removeCreator,
    })),
  );

  const handleFilterClick = (filter: CategoryKey) => {
    if (filters.includes(filter)) {
      removeFilter(filter);
    } else {
      addFilter(filter);
    }
  };

  const handleCreatorClick = (creator: string) => {
    if (creators.includes(creator)) {
      removeCreator(creator);
    } else {
      addCreator(creator);
    }
  };

  return (
    <div className="flex flex-wrap gap-2">
      <FilterSheet categories={categories} />
      {creators.length > 0 &&
        creators.map((creator) => (
          <FilterChip
            key={creator}
            name={"Created by " + creator.slice(0, 10) + "..."}
            selected={creators.includes(creator)}
            onClick={() => handleCreatorClick(creator)}
          />
        ))}
      {categories.map((category) => (
        <FilterChip
          key={category.key}
          name={category.name}
          selected={filters.includes(category.key)}
          onClick={() => handleFilterClick(category.key)}
          number={categoryCounts[category.key] ?? 0}
        />
      ))}
    </div>
  );
};
