"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { Bell, MagnifyingGlass, X } from "@phosphor-icons/react";
import { FixedSizeList as List } from "react-window";
import { AgentExecutionWithInfo } from "../../helpers";
import { ActivityItem } from "../ActivityItem";
import styles from "./styles.module.css";
import {
  EXECUTION_DISPLAY_WITH_SEARCH,
  useActivityDropdown,
} from "./useActivityDropdown";

interface Props {
  activeExecutions: AgentExecutionWithInfo[];
  recentCompletions: AgentExecutionWithInfo[];
  recentFailures: AgentExecutionWithInfo[];
}

interface VirtualizedItemProps {
  index: number;
  style: React.CSSProperties;
  data: AgentExecutionWithInfo[];
}

function VirtualizedActivityItem({ index, style, data }: VirtualizedItemProps) {
  const execution = data[index];
  return (
    <div style={style}>
      <ActivityItem execution={execution} />
    </div>
  );
}

export function ActivityDropdown({
  activeExecutions,
  recentCompletions,
  recentFailures,
}: Props) {
  const {
    isSearchVisible,
    searchQuery,
    filteredExecutions,
    toggleSearch,
    totalExecutions,
    handleSearchChange,
    handleClearSearch,
  } = useActivityDropdown({
    activeExecutions,
    recentCompletions,
    recentFailures,
  });

  // Static height for the virtualised list (react-window)
  const itemHeight = 72; // Height of each ActivityItem in pixels
  const maxHeight = 400; // Maximum height of the dropdown

  const listHeight = Math.min(
    maxHeight,
    filteredExecutions.length * itemHeight,
  );

  const withSearch = totalExecutions > EXECUTION_DISPLAY_WITH_SEARCH;

  return (
    <div className="overflow-hidden">
      {/* Header */}
      <div className="sticky top-0 z-10 px-4 pb-1 pt-0">
        <div className="flex h-[60px] items-center justify-between">
          {isSearchVisible && withSearch ? (
            <div
              className={`${styles.searchContainer} ${
                isSearchVisible ? styles.searchEnter : styles.searchExit
              }`}
            >
              <div className="relative w-full">
                <Input
                  id="agent-search"
                  label="Search agents"
                  placeholder="Search runs by agent name..."
                  hideLabel
                  size="small"
                  value={searchQuery}
                  onChange={(e) => handleSearchChange(e.target.value)}
                  className="!focus:border-1 w-full pr-10"
                  wrapperClassName="!mb-0"
                  autoComplete="off"
                  autoFocus
                />
                <button
                  onClick={handleClearSearch}
                  className="absolute right-1 top-1/2 flex h-6 w-6 -translate-y-1/2 items-center justify-center"
                  aria-label="Clear search"
                >
                  <X size={16} className="text-gray-500" />
                </button>
              </div>
            </div>
          ) : (
            <div className={styles.headerContainer}>
              <Text variant="large-semibold" className="!text-black">
                Agent Activity
              </Text>
              {withSearch ? (
                <Button
                  variant="ghost"
                  size="small"
                  onClick={toggleSearch}
                  aria-label="Search agents"
                  className="relative left-3 hover:border-transparent hover:bg-transparent"
                >
                  <MagnifyingGlass
                    size={16}
                    className="h-4 w-4 text-gray-600"
                  />
                </Button>
              ) : null}
            </div>
          )}
        </div>
      </div>

      {/* Content */}
      <div
        className={styles.scrollContainer}
        data-testid="agent-activity-dropdown"
      >
        {filteredExecutions.length > 0 ? (
          <List
            height={listHeight}
            width={320} // Match dropdown width (w-80 = 20rem = 320px)
            itemCount={filteredExecutions.length}
            itemSize={itemHeight}
            itemData={filteredExecutions}
          >
            {VirtualizedActivityItem}
          </List>
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-5 pb-8 pt-6">
            <div className="mx-auto inline-flex flex-col items-center justify-center rounded-full bg-bgLightGrey p-6">
              <Bell className="h-6 w-6 text-zinc-300" />
            </div>
            <div className="flex flex-col items-center justify-center">
              <Text variant="body-medium" className="!text-black">
                {searchQuery
                  ? "No matching agents found"
                  : "No recent runs to show yet"}
              </Text>
              <Text variant="body" className="!text-zinc-500">
                {searchQuery
                  ? "Try another search term"
                  : "Start an agent to get updates"}
              </Text>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
