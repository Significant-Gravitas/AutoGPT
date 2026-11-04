import { cn } from "@/lib/utils";
import type { ReactNode } from "react";
import { describeLiveCounts } from "../../helpers";
import type { useTasksPage } from "../../useTasksPage";
import { InboxBoard } from "../InboxBoard/InboxBoard";
import { InboxEmpty } from "../InboxEmpty/InboxEmpty";
import { InboxError } from "../InboxEmpty/InboxError";
import { InboxFilters } from "../InboxFilters/InboxFilters";
import { InboxHeader } from "../InboxHeader/InboxHeader";
import { InboxList } from "../InboxList/InboxList";
import { InboxListSkeleton } from "../InboxList/InboxListSkeleton";
import { InboxToolbar } from "../InboxToolbar/InboxToolbar";
import { InboxCard } from "../InboxList/InboxCard";
import { LiveRegion } from "../LiveRegion/LiveRegion";

type Page = ReturnType<typeof useTasksPage>;

interface Props {
  page: Page;
  /** Home's Chat / Board switch. Given, the pane is board-only and its
   *  controls collapse into one toolbar around the switch. */
  viewSwitch?: ReactNode;
}

export function InboxPane({ page, viewSwitch }: Props) {
  // The board scrolls sideways, so it works at any width.
  const isBoard = viewSwitch != null || page.view === "board";

  return (
    <div className="flex h-full min-h-0 flex-col">
      {viewSwitch != null ? (
        <InboxToolbar page={page} viewSwitch={viewSwitch} />
      ) : (
        <div className="mx-auto flex w-full max-w-6xl shrink-0 flex-col gap-4 pb-4">
          <InboxHeader
            view={page.view}
            onViewChange={page.setView}
            onNewTask={() => page.setIsCreateOpen(true)}
          />
          <InboxFilters
            tab={page.tab}
            onTabChange={page.setTab}
            counts={page.counts}
            filters={page.filters}
            onFiltersChange={page.setFilters}
            chips={page.filterChips}
            experts={page.experts}
          />
        </div>
      )}
      <LiveRegion message={describeLiveCounts(page.counts)} />
      <div className="-mx-1 min-h-0 flex-1 overflow-y-auto px-1 pb-1">
        <div className={cn("mx-auto w-full max-w-6xl", isBoard && "h-full")}>
          <InboxBody page={page} isBoard={isBoard} />
        </div>
      </div>
    </div>
  );
}

interface BodyProps {
  page: Page;
  isBoard: boolean;
}

function InboxBody({ page, isBoard }: BodyProps) {
  if (page.error && !page.hasData) {
    return (
      <InboxCard>
        <InboxError onRetry={() => page.refetch()} />
      </InboxCard>
    );
  }
  if (page.isLoading) return <InboxListSkeleton />;
  if (page.tasks.length === 0) {
    return (
      <InboxCard>
        <InboxEmpty
          tab={page.tab}
          isNewUser={page.isNewUser}
          hasFilters={page.filterChips.length > 0}
          onClearFilters={page.clearFilters}
          onNewTask={() => page.setIsCreateOpen(true)}
        />
      </InboxCard>
    );
  }
  if (isBoard) {
    return (
      <InboxBoard
        tasks={page.tasks}
        selectedID={page.taskID}
        onOpen={page.openTask}
      />
    );
  }
  return (
    <InboxList
      key={page.tab}
      tasks={page.tasks}
      selectedID={page.taskID}
      onOpen={page.openTask}
    />
  );
}
