import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { SegmentedControl } from "@/components/molecules/SegmentedControl/SegmentedControl";
import { UserIcon } from "@hugeicons/core-free-icons";
import type { ExpertOption, TaskCounts } from "../../data/types";
import {
  INBOX_TAB_LABELS,
  INBOX_TABS,
  tabCount,
  type FilterChip,
  type InboxTab,
  type PageFilters,
} from "../../helpers";
import { FilterMenu } from "../FilterMenu/FilterMenu";
import { ActiveFilters } from "./ActiveFilters";

interface Props {
  tab: InboxTab;
  onTabChange: (tab: InboxTab) => void;
  counts: TaskCounts | undefined;
  mine: boolean;
  onMineChange: (mine: boolean) => void;
  filters: PageFilters;
  onFiltersChange: (filters: PageFilters) => void;
  chips: FilterChip[];
  experts: ExpertOption[];
}

export function InboxFilters({
  tab,
  onTabChange,
  counts,
  mine,
  onMineChange,
  filters,
  onFiltersChange,
  chips,
  experts,
}: Props) {
  const options = INBOX_TABS.map((value) => ({
    value,
    label: INBOX_TAB_LABELS[value],
    count: tabCount(counts, value),
  }));

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="-mx-1 scrollbar-none max-w-full overflow-x-auto px-1">
          <SegmentedControl
            size="sm"
            value={tab}
            onChange={onTabChange}
            options={options}
            ariaLabel="Show tasks"
          />
        </div>
        <div className="flex items-center gap-1">
          <Button
            variant="toggle"
            size="xs"
            aria-pressed={mine}
            leftIcon={<Icon icon={UserIcon} size={14} />}
            onClick={() => onMineChange(!mine)}
          >
            Mine
          </Button>
          <FilterMenu
            filters={filters}
            onChange={onFiltersChange}
            experts={experts}
          />
        </div>
      </div>
      <ActiveFilters chips={chips} onChange={onFiltersChange} />
    </div>
  );
}
