import { LibraryAgentSort } from "@/app/api/__generated__/models/libraryAgentSort";

interface Props {
  setLibrarySort: (value: LibraryAgentSort) => void;
}

export function useLibrarySortMenu({ setLibrarySort }: Props) {
  const handleSortChange = (value: string) => {
    setLibrarySort(value as LibraryAgentSort);
  };

  const getSortLabel = (sort: LibraryAgentSort) => {
    switch (sort) {
      case LibraryAgentSort.createdAt:
        return "Creation Date";
      case LibraryAgentSort.updatedAt:
        return "Last Modified";
      case LibraryAgentSort.lastRunAt:
        return "Last Run";
      default:
        return "Last Modified";
    }
  };

  return {
    handleSortChange,
    getSortLabel,
  };
}
