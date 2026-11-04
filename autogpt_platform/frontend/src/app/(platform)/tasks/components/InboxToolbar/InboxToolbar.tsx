import type { ReactNode } from "react";
import type { useTasksPage } from "../../useTasksPage";
import { FilterMenu } from "../FilterMenu/FilterMenu";
import { ActiveFilters } from "../InboxFilters/ActiveFilters";
import { StatusTabs } from "../InboxFilters/StatusTabs";
import { NewTaskButton } from "../NewTaskButton/NewTaskButton";

interface Props {
  page: ReturnType<typeof useTasksPage>;
  viewSwitch: ReactNode;
}

// One row when there is room: status tabs on the leading side, the view
// switch centred where the chat view puts it, filter and New task trailing.
// Narrower, the tabs drop to their own row; narrower still, the switch leads.
export function InboxToolbar({ page, viewSwitch }: Props) {
  return (
    <div className="@container mx-auto flex w-full max-w-6xl shrink-0 flex-col gap-2 pt-4 pb-4 md:pt-6">
      <div className="grid grid-cols-[auto_minmax(0,1fr)] items-center gap-3 @lg:grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)]">
        <div className="col-span-full row-start-2 min-w-0 @5xl:col-span-1 @5xl:col-start-1 @5xl:row-start-1">
          <StatusTabs
            tab={page.tab}
            onTabChange={page.setTab}
            counts={page.counts}
          />
        </div>
        <div className="col-start-1 row-start-1 @lg:col-start-2">
          {viewSwitch}
        </div>
        <div className="col-start-2 row-start-1 flex items-center justify-end gap-2 @lg:col-start-3">
          <FilterMenu
            filters={page.filters}
            onChange={page.setFilters}
            experts={page.experts}
          />
          <NewTaskButton onClick={() => page.setIsCreateOpen(true)} />
        </div>
      </div>
      <ActiveFilters chips={page.filterChips} onChange={page.setFilters} />
    </div>
  );
}
