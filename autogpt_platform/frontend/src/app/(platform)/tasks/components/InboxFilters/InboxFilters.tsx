import type { ExpertOption, TaskCounts } from "../../data/types";
import type { FilterChip, InboxTab, PageFilters } from "../../helpers";
import { FilterMenu } from "../FilterMenu/FilterMenu";
import { ActiveFilters } from "./ActiveFilters";
import { StatusTabs } from "./StatusTabs";

interface Props {
  tab: InboxTab;
  onTabChange: (tab: InboxTab) => void;
  counts: TaskCounts | undefined;
  filters: PageFilters;
  onFiltersChange: (filters: PageFilters) => void;
  chips: FilterChip[];
  experts: ExpertOption[];
}

export function InboxFilters({
  tab,
  onTabChange,
  counts,
  filters,
  onFiltersChange,
  chips,
  experts,
}: Props) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <StatusTabs tab={tab} onTabChange={onTabChange} counts={counts} />
        <FilterMenu
          filters={filters}
          onChange={onFiltersChange}
          experts={experts}
        />
      </div>
      <ActiveFilters chips={chips} onChange={onFiltersChange} />
    </div>
  );
}
