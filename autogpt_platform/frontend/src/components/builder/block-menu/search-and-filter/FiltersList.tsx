import { useState, useEffect } from "react";
import FilterChip from "../FilterChip";
import FilterSheet from "./FilterSheet";

export type CategoryKey =
  | "blocks"
  | "integrations"
  | "marketplace_agents"
  | "my_agents"
  | "templates";

export interface Filters {
  categories: {
    blocks: boolean;
    integrations: boolean;
    marketplace_agents: boolean;
    my_agents: boolean;
    templates: boolean;
  };
  createdBy: string[];
}

const FiltersList = () => {
  const [filters, setFilters] = useState<Filters>({
    categories: {
      blocks: false,
      integrations: false,
      marketplace_agents: false,
      my_agents: false,
      templates: false,
    },
    createdBy: [],
  });

  const [creators, setCreators] = useState<string[]>([]);

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
  }, []);

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
    <div className="flex flex-wrap gap-3">
      <FilterSheet
        filters={filters}
        creators={creators}
        categories={categories}
        onCategoryChange={handleCategoryFilter}
        onCreatorChange={handleCreatorFilter}
      />

      {filters.createdBy.map((creator) => (
        <FilterChip
          key={creator}
          name={creator}
          selected={true}
          number={4}
          onClick={() => handleCreatorFilter(creator)}
        />
      ))}

      {categories.map((category) => (
        <FilterChip
          key={category.key}
          name={category.name}
          selected={filters.categories[category.key]}
          onClick={() => handleCategoryFilter(category.key)}
        />
      ))}
    </div>
  );
};

export default FiltersList;
