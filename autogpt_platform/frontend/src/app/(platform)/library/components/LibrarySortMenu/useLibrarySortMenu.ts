import { useCallback } from "react";
import { LibraryAgentSort } from "@/app/api/__generated__/models/libraryAgentSort";
import { useLibraryPageContext } from "../state-provider";

export const useLibrarySortMenu = () => {
  const { setLibrarySort } = useLibraryPageContext();

  const getSortLabel = useCallback((sort: LibraryAgentSort) => {
    return {
      [LibraryAgentSort.createdAt]: "Creation Date",
      [LibraryAgentSort.updatedAt]: "Last Modified",
      [LibraryAgentSort.lastExecuted]: "Last Executed",
    }[sort];
  }, []);

  return {
    setLibrarySort,
    getSortLabel,
  };
};
