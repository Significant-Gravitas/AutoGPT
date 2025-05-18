import React from "react";
import FiltersList from "./FiltersList";
import SearchList from "./SearchList.";

const BlockMenuSearch: React.FC<{ searchQuery: string }> = ({
  searchQuery,
}) => {
  return (
    <div className="scrollbar-thumb-rounded h-full space-y-4 overflow-y-scroll p-4 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-200">
      <FiltersList />
      <SearchList searchQuery={searchQuery} />
    </div>
  );
};

export default BlockMenuSearch;
