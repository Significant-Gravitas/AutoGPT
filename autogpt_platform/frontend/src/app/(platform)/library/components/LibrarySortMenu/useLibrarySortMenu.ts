import { LibraryAgentSortEnum as LibraryAgentSort } from "@/lib/autogpt-server-api/types";
import { useLibraryPageContext } from "../state-provider";

export const useLibrarySortMenu = () => {
  const { setLibrarySort } = useLibraryPageContext();

  const handleSortChange = (value: LibraryAgentSort) => {
    // Simply updating the sort state - React Query will handle the rest
    setLibrarySort(value);
  };

  const getSortLabel = (sort: LibraryAgentSort) => {
    switch (sort) {
      case LibraryAgentSort.CREATED_AT:
        return "Creation Date";
      case LibraryAgentSort.UPDATED_AT:
        return "Last Modified";
      case LibraryAgentSort.FAVORITES_FIRST:
        return "Favorites First";
      default:
        return "Favorites First";
    }
  };

  return {
    handleSortChange,
    getSortLabel,
  };
};
