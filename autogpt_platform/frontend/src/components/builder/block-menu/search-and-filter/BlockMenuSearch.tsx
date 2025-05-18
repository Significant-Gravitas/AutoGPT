import React from "react";
import FiltersList from "./FiltersList";
import SearchList from "./SearchList.";

const BlockMenuSearch: React.FC<{ searchQuery: string }> = ({
  searchQuery,
}) => {
  return (
    <div className="scrollbar-thumb-rounded scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-200 h-full space-y-4 overflow-y-scroll p-4">
      <FiltersList />
      <SearchList searchQuery={searchQuery} />
    </div>
  );
};

export default BlockMenuSearch;
