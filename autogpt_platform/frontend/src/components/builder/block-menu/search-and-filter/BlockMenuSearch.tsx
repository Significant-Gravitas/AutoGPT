import React from "react";
import FiltersList from "./FiltersList";
import SearchList from "./SearchList";
import { useBlockMenuContext } from "../block-menu-provider";

const BlockMenuSearch: React.FC = ({}) => {
  const { searchData } = useBlockMenuContext();

  return (
    <div className="scrollbar-thumb-rounded h-full space-y-4 overflow-y-auto py-4 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-200">
      {searchData.length !== 0 && <FiltersList />}
      <SearchList />
    </div>
  );
};

export default BlockMenuSearch;
