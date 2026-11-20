"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { List, type RowComponentProps } from "react-window";
import { AgentExecutionWithInfo } from "../../helpers";
import { ActivityItem } from "../ActivityItem";
import styles from "./styles.module.css";
import {
  EXECUTION_DISPLAY_WITH_SEARCH,
  useActivityDropdown,
} from "./useActivityDropdown";
import {
  BellIcon,
  Cancel01Icon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  activeExecutions: AgentExecutionWithInfo[];
  recentCompletions: AgentExecutionWithInfo[];
  recentFailures: AgentExecutionWithInfo[];
}

interface ActivityRowProps {
  executions: AgentExecutionWithInfo[];
}

function VirtualizedActivityItem({
  index,
  style,
  executions,
}: RowComponentProps<ActivityRowProps>) {
  const execution = executions[index];
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
  const maxHeight = 320; // Maximum height of the dropdown

  const listHeight = Math.min(
    maxHeight,
    filteredExecutions.length * itemHeight,
  );

  const withSearch = totalExecutions > EXECUTION_DISPLAY_WITH_SEARCH;

  return (
    <div className="overflow-hidden">
      {/* Header */}
      <div className="sticky top-0 z-10 px-4">
        <div className="flex items-center justify-between pt-3 pb-1">
          {isSearchVisible && withSearch ? (
            <div
              className={`${styles.searchContainer} ${
                isSearchVisible ? styles.searchEnter : styles.searchExit
              }`}
            >
              <div className="relative w-full">
                <Input
                  id="agent-search"
                  label="Search workflows"
                  placeholder="Search runs by workflow name..."
                  hideLabel
                  size="small"
                  value={searchQuery}
                  onChange={(e) => handleSearchChange(e.target.value)}
                  className="w-full pr-10"
                  wrapperClassName="!mb-0"
                  autoComplete="off"
                  autoFocus
                />
                <button
                  onClick={handleClearSearch}
                  className="absolute top-1/2 right-1 flex h-6 w-6 -translate-y-1/2 items-center justify-center"
                  aria-label="Clear search"
                >
                  <Icon icon={Cancel01Icon} size={16} tone="muted" />
                </button>
              </div>
            </div>
          ) : (
            <div className={`${styles.headerContainer} py-0.5`}>
              <span className="text-caption font-medium text-foreground-tertiary uppercase">
                Workflow Activity
              </span>
              {withSearch ? (
                <Button
                  variant="toggle"
                  size="inline"
                  onClick={toggleSearch}
                  aria-label="Search workflows"
                  className="w-fit"
                >
                  <Icon
                    icon={Search01Icon}
                    size={16}
                    tone="secondary"
                    className="h-4 w-4"
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
            defaultHeight={listHeight}
            rowCount={filteredExecutions.length}
            rowHeight={itemHeight}
            rowProps={{ executions: filteredExecutions }}
            rowComponent={VirtualizedActivityItem}
            style={{ width: "100%", height: listHeight }}
          />
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-5 pt-6 pb-8">
            <div className="mx-auto inline-flex flex-col items-center justify-center rounded-full bg-surface-hover p-6">
              <Icon icon={BellIcon} tone="disabled" className="h-6 w-6" />
            </div>
            <div className="flex flex-col items-center justify-center">
              <Text variant="body-medium">
                {searchQuery
                  ? "No matching workflows found"
                  : "No recent runs to show yet"}
              </Text>
              <Text variant="body" tone="muted">
                {searchQuery
                  ? "Try another search term"
                  : "Start a workflow to get updates"}
              </Text>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
