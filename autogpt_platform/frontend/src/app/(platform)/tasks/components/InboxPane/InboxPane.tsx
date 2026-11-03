import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { describeLiveCounts } from "../../helpers";
import { useContainerWidth } from "../../useContainerWidth";
import type { useTasksPage } from "../../useTasksPage";
import { InboxBoard } from "../InboxBoard/InboxBoard";
import { InboxEmpty } from "../InboxEmpty/InboxEmpty";
import { InboxFilters } from "../InboxFilters/InboxFilters";
import { InboxHeader } from "../InboxHeader/InboxHeader";
import { InboxList } from "../InboxList/InboxList";
import { InboxListSkeleton } from "../InboxList/InboxListSkeleton";
import { InboxTray } from "../InboxList/InboxTray";
import { LiveRegion } from "../LiveRegion/LiveRegion";

type Page = ReturnType<typeof useTasksPage>;

interface Props {
  page: Page;
}

export function InboxPane({ page }: Props) {
  const { attach, isWide } = useContainerWidth();

  return (
    <div ref={attach} className="flex h-full min-h-0 flex-col">
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
          mine={page.mine}
          onMineChange={page.setMine}
          filters={page.filters}
          onFiltersChange={page.setFilters}
          chips={page.filterChips}
          experts={page.experts}
        />
        <LiveRegion message={describeLiveCounts(page.counts)} />
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-6xl">
          <InboxBody page={page} isBoard={page.view === "board" && isWide} />
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
  if (page.error) {
    return (
      <ErrorCard
        context="tasks"
        responseError={{ message: page.error.message }}
        onRetry={() => page.refetch()}
      />
    );
  }
  if (page.isLoading) return <InboxListSkeleton />;
  if (page.tasks.length === 0) {
    return (
      <InboxTray>
        <InboxEmpty
          tab={page.tab}
          hasFilters={page.filterChips.length > 0 || page.mine}
          onClearFilters={page.clearFilters}
        />
      </InboxTray>
    );
  }
  if (isBoard) {
    return (
      <InboxBoard
        tasks={page.tasks}
        tab={page.tab}
        expandedID={page.expandedID}
        selectedID={page.taskID}
        onToggle={page.toggleRow}
        onOpen={page.openTask}
      />
    );
  }
  return (
    <InboxList
      tasks={page.tasks}
      expandedID={page.expandedID}
      selectedID={page.taskID}
      onToggle={page.toggleRow}
      onOpen={page.openTask}
    />
  );
}
