import { LibraryAgentSort } from "@/app/api/__generated__/models/libraryAgentSort";

interface Props {
  setLibrarySort: (value: LibraryAgentSort) => void;
}

export function useLibrarySortMenu({ setLibrarySort }: Props) {
  const handleSortChange = (value: LibraryAgentSort) => {
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
}
