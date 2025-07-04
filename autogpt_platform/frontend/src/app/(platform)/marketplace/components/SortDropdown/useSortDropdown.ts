import { useState } from "react";
import { SortOption } from "./SortDropdown";

interface useSortDropdownProps {
  onSort: (sortValue: string) => void;
  sortOptions: SortOption[];
}

export const useSortDropdown = ({
  onSort,
  sortOptions,
}: useSortDropdownProps) => {
  const [selected, setSelected] = useState(sortOptions[0]);

  const handleSelect = (option: SortOption) => {
    setSelected(option);
    onSort(option.value);
  };

  return {
    selected,
    handleSelect,
  };
};
