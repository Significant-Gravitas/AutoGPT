import { LibraryAgentSort } from "@/app/api/__generated__/models/libraryAgentSort";
import { useLibraryPageContext } from "../state-provider";

export const useLibrarySortMenu = () => {
  const { setLibrarySort } = useLibraryPageContext();

  const handleSortChange = (value: LibraryAgentSort) => {
    // Simply updating the sort state - React Query will handle the rest
    setLibrarySort(value);
  };

  const getSortLabel = (sort: LibraryAgentSort) => {
    switch (sort) {
      case LibraryAgentSort.createdAt:
        return "Creation Date";
      case LibraryAgentSort.updatedAt:
        return "Last Modified";
      default:
        return "Last Modified";
    }
  };

  return {
    handleSortChange,
    getSortLabel,
  };
};
