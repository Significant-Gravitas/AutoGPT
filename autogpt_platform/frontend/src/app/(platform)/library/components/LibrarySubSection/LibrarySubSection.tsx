import { useEffect, useRef } from "react";
import {
  TabsLine,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";
import { LibraryAgentSort } from "@/app/api/__generated__/models/libraryAgentSort";
import { useFavoriteAnimation } from "../../context/FavoriteAnimationContext";
import type { LibraryTab, AgentStatusFilter, FleetSummary } from "../../types";
import LibraryFolderCreationDialog from "../LibraryFolderCreationDialog/LibraryFolderCreationDialog";
import { LibrarySortMenu } from "../LibrarySortMenu/LibrarySortMenu";
import { AgentFilterMenu } from "../AgentFilterMenu/AgentFilterMenu";

interface Props {
  tabs: LibraryTab[];
  activeTab: string;
  onTabChange: (tabId: string) => void;
  allCount: number;
  favoritesCount: number;
  setLibrarySort: (value: LibraryAgentSort) => void;
  statusFilter?: AgentStatusFilter;
  onStatusFilterChange?: (filter: AgentStatusFilter) => void;
  fleetSummary?: FleetSummary;
}

export function LibrarySubSection({
  tabs,
  activeTab,
  onTabChange,
  allCount,
  favoritesCount,
  setLibrarySort,
  statusFilter = "all",
  onStatusFilterChange,
  fleetSummary,
}: Props) {
  const { registerFavoritesTabRef } = useFavoriteAnimation();
  const favoritesRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    registerFavoritesTabRef(favoritesRef.current);
    return () => {
      registerFavoritesTabRef(null);
    };
  }, [registerFavoritesTabRef]);

  function getTabLabel(tab: LibraryTab) {
    if (tab.id === "all") {
      return `${tab.title} ${allCount}`;
    }
    if (tab.id === "favorites") {
      return favoritesCount > 0 ? `${tab.title} ${favoritesCount}` : tab.title;
    }
    return tab.title;
  }

  return (
    <div className="flex items-center justify-between gap-4">
      <span data-testid="agents-count" className="sr-only">
        {allCount}
      </span>
      <TabsLine value={activeTab} onValueChange={onTabChange}>
        <TabsLineList>
          {tabs.map((tab) => (
            <TabsLineTrigger
              key={tab.id}
              value={tab.id}
              ref={tab.id === "favorites" ? favoritesRef : undefined}
              className="inline-flex items-center gap-1.5"
              disabled={tab.id === "favorites" && favoritesCount === 0}
            >
              <tab.icon size={16} />
              {getTabLabel(tab)}
            </TabsLineTrigger>
          ))}
        </TabsLineList>
      </TabsLine>
      <div className="relative top-1.5 hidden items-center gap-6 md:flex">
        <LibraryFolderCreationDialog />
        {fleetSummary && onStatusFilterChange && (
          <AgentFilterMenu
            value={statusFilter}
            onChange={onStatusFilterChange}
            summary={fleetSummary}
          />
        )}
        <LibrarySortMenu setLibrarySort={setLibrarySort} />
      </div>
    </div>
  );
}
