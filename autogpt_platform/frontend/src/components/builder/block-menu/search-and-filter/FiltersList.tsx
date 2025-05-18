import FilterChip from "../FilterChip";

const FiltersList = () => {
  return (
    <div className="space-x-3">
      <FilterChip name="All filters" />
      {/* Created by filters */}

      {/* Fixed filters */}
      <FilterChip name="Blocks" />
      <FilterChip name="Integrations" />
      <FilterChip name="Marketplace agents" />
      <FilterChip name="My agents" />
    </div>
  );
};

export default FiltersList;
