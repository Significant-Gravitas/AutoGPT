import React from "react";
import FiltersList from "./FiltersList";
import SearchList from "./SearchList.";

const BlockMenuSearch: React.FC = ({}) => {
  return (
    <div className="scrollbar-thumb-rounded h-full space-y-4 overflow-y-auto p-4 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-200">
      <FiltersList />
      <SearchList />
    </div>
  );
};

export default BlockMenuSearch;
