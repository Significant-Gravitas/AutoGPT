import { useState, useEffect } from "react";
import FilterChip from "../FilterChip";
import FilterSheet from "./FilterSheet";
import { CategoryKey, useBlockMenuContext } from "../block-menu-provider";

const FiltersList = () => {
  const { setCreators, filters, setFilters } = useBlockMenuContext();
  const categories: Array<{ key: CategoryKey; name: string }> = [
    { key: "blocks", name: "Blocks" },
    { key: "integrations", name: "Integrations" },
    { key: "marketplace_agents", name: "Marketplace agents" },
    { key: "my_agents", name: "My agents" },
    { key: "templates", name: "Templates" },
  ];

  // TEMPORARY FETCHING
  useEffect(() => {
    const mockCreators = ["Abhi", "Abhi 1", "Abhi 2", "Abhi 3", "Abhi 4"];
    setCreators(mockCreators);
  }, [setCreators]);

  const handleCategoryFilter = (category: CategoryKey) => {
    setFilters({
      ...filters,
      categories: {
        ...filters.categories,
        [category]: !filters.categories[category],
      },
    });
  };

  const handleCreatorFilter = (creator: string) => {
    const updatedCreators = filters.createdBy.includes(creator)
      ? filters.createdBy.filter((c) => c !== creator)
      : [...filters.createdBy, creator];

    setFilters({
      ...filters,
      createdBy: updatedCreators,
    });
  };

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
          number={103}
          selected={filters.categories[category.key]}
          onClick={() => handleCategoryFilter(category.key)}
        />
      ))}
    </div>
  );
};

export default FiltersList;
