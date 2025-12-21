import { useBlockMenuStore } from "@/app/(platform)/build/stores/blockMenuStore";
import { FilterChip } from "../FilterChip";
import { categories } from "./constants";
import { GetV2BuilderSearchFilterAnyOfItem } from "@/app/api/__generated__/models/getV2BuilderSearchFilterAnyOfItem";

export const BlockMenuFilters = () => {
  const { filters, setFilter, removeFilter } = useBlockMenuStore();

  const handleFilterClick = (filter: GetV2BuilderSearchFilterAnyOfItem) => {
    if (filters.includes(filter)) {
      removeFilter(filter);
    } else {
      setFilter(filter);
    }
  };

  return (
    <div className="flex flex-wrap gap-2">
      {categories.map((category) => (
        <FilterChip
          key={category.key}
          name={category.name}
          selected={filters.includes(category.key)}
          onClick={() => handleFilterClick(category.key)}
        />
      ))}
    </div>
  );
};
