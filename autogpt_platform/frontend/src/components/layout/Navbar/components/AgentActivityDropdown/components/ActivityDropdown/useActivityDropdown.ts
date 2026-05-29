import { useState, useMemo } from "react";
import { AgentExecutionWithInfo } from "../../helpers";
import { getSortedExecutions } from "./helpers";

export const EXECUTION_DISPLAY_WITH_SEARCH = 6;

interface UseActivityDropdownProps {
  activeExecutions: AgentExecutionWithInfo[];
  recentCompletions: AgentExecutionWithInfo[];
  recentFailures: AgentExecutionWithInfo[];
}

export function useActivityDropdown({
  activeExecutions,
  recentCompletions,
  recentFailures,
}: UseActivityDropdownProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [isSearchVisible, setIsSearchVisible] = useState(false);

  const sortedExecutions = getSortedExecutions({
    activeExecutions,
    recentCompletions,
    recentFailures,
  });

  // Filter executions based on search query
  const filteredExecutions = useMemo(() => {
    if (!searchQuery.trim()) {
      return sortedExecutions;
    }

    const query = searchQuery.toLowerCase().trim();
    return sortedExecutions.filter((execution) =>
      execution.agent_name.toLowerCase().includes(query),
    );
  }, [sortedExecutions, searchQuery]);

  function toggleSearch() {
    setIsSearchVisible(!isSearchVisible);
    if (searchQuery) {
      setSearchQuery("");
    }
  }

  function handleSearchChange(value: string) {
    setSearchQuery(value);
  }

  function handleClearSearch() {
    handleSearchChange("");
    toggleSearch();
  }

  function handleShowSearch() {
    setIsSearchVisible(true);
  }

  return {
    isSearchVisible,
    searchQuery,
    filteredExecutions,
    totalExecutions: sortedExecutions.length,
    toggleSearch,
    handleSearchChange,
    handleClearSearch,
    handleShowSearch,
  };
}
