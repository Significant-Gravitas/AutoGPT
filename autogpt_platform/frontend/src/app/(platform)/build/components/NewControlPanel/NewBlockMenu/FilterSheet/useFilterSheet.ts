import { useBlockMenuStore } from "@/app/(platform)/build/stores/blockMenuStore";
import { useState } from "react";
import { INITIAL_CREATORS_TO_SHOW } from "./constant";
import { GetV2BuilderSearchFilterAnyOfItem } from "@/app/api/__generated__/models/getV2BuilderSearchFilterAnyOfItem";

export const useFilterSheet = () => {
  const { filters, creators_list, creators, setFilters, setCreators } =
    useBlockMenuStore();

  const [isOpen, setIsOpen] = useState(false);
  const [localCategories, setLocalCategories] =
    useState<GetV2BuilderSearchFilterAnyOfItem[]>(filters);
  const [localCreators, setLocalCreators] = useState<string[]>(creators);
  const [displayedCreatorsCount, setDisplayedCreatorsCount] = useState(
    INITIAL_CREATORS_TO_SHOW,
  );

  const handleLocalCategoryChange = (
    category: GetV2BuilderSearchFilterAnyOfItem,
  ) => {
    setLocalCategories((prev) => {
      if (prev.includes(category)) {
        return prev.filter((c) => c !== category);
      }
      return [...prev, category];
    });
  };

  const hasActiveFilters = () => {
    return filters.length > 0 || creators.length > 0;
  };

  const handleToggleShowMoreCreators = () => {
    if (displayedCreatorsCount < creators.length) {
      setDisplayedCreatorsCount(creators.length);
    } else {
      setDisplayedCreatorsCount(INITIAL_CREATORS_TO_SHOW);
    }
  };

  const handleLocalCreatorChange = (creator: string) => {
    setLocalCreators((prev) => {
      if (prev.includes(creator)) {
        return prev.filter((c) => c !== creator);
      }
      return [...prev, creator];
    });
  };

  const handleClearFilters = () => {
    setLocalCategories([]);
    setLocalCreators([]);
    setDisplayedCreatorsCount(INITIAL_CREATORS_TO_SHOW);
  };

  const handleCloseButton = () => {
    setIsOpen(false);
    setLocalCategories(filters);
    setLocalCreators(creators);
    setDisplayedCreatorsCount(INITIAL_CREATORS_TO_SHOW);
  };

  const handleApplyFilters = () => {
    setFilters(localCategories);
    setCreators(localCreators);
    setIsOpen(false);
  };

  const handleOpenFilters = () => {
    setIsOpen(true);
    setLocalCategories(filters);
    setLocalCreators(creators);
  };

  const hasLocalActiveFilters = () => {
    return localCategories.length > 0 || localCreators.length > 0;
  };

  const visibleCreators = creators_list.slice(0, displayedCreatorsCount);

  return {
    creators,
    isOpen,
    setIsOpen,
    localCategories,
    localCreators,
    displayedCreatorsCount,
    setDisplayedCreatorsCount,
    handleLocalCategoryChange,
    handleToggleShowMoreCreators,
    handleLocalCreatorChange,
    handleClearFilters,
    handleCloseButton,
    handleOpenFilters,
    handleApplyFilters,
    hasLocalActiveFilters,
    visibleCreators,
    hasActiveFilters,
  };
};
