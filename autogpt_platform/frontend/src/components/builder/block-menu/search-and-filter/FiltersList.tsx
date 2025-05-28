import { useState, useEffect, useCallback } from "react";
import FilterChip from "../FilterChip";
import FilterSheet from "./FilterSheet";
import { CategoryKey, useBlockMenuContext } from "../block-menu-provider";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";

const FiltersList = () => {
  const { filters, setFilters, categoryCounts, setCategoryCounts } =
    useBlockMenuContext();
  const api = useBackendAPI();
  const categories: Array<{ key: CategoryKey; name: string }> = [
    { key: "blocks", name: "Blocks" },
    { key: "integrations", name: "Integrations" },
    { key: "marketplace_agents", name: "Marketplace agents" },
    { key: "my_agents", name: "My agents" },
    { key: "providers", name: "Providers" },
  ];

  const handleCategoryFilter = (category: CategoryKey) => {
    setFilters({
      ...filters,
      categories: {
        ...filters.categories,
        [category]: !filters.categories[category],
      },
    });
  };

  const handleCreatorFilter = useCallback(
    (creator: string) => {
      const updatedCreators = filters.createdBy.includes(creator)
        ? filters.createdBy.filter((c) => c !== creator)
        : [...filters.createdBy, creator];

      setFilters({
        ...filters,
        createdBy: updatedCreators,
      });
    },
    [filters, setFilters],
  );

  useEffect(() => {
    console.log(categoryCounts);
  }, [categoryCounts]);

  return (
    <div className="flex flex-nowrap gap-3 overflow-x-auto scrollbar-hide">
      <FilterSheet categories={categories} />

      {filters.createdBy.map((creator) => (
        <FilterChip
          key={creator}
          name={"Created by " + creator}
          selected={true}
          onClick={() => handleCreatorFilter(creator)}
        />
      ))}

      {categories.map((category) => (
        <FilterChip
          key={category.key}
          name={category.name}
          needHover={
            Object.values(filters.categories).filter(Boolean).length === 1 &&
            filters.categories[category.key]
          }
          number={categoryCounts[category.key]}
          selected={filters.categories[category.key]}
          onClick={() => handleCategoryFilter(category.key)}
        />
      ))}
    </div>
  );
};

export default FiltersList;
