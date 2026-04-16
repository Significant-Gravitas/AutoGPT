"use client";

import { LibraryAgentSort } from "@/app/api/__generated__/models/libraryAgentSort";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { InfiniteScroll } from "@/components/contextual/InfiniteScroll/InfiniteScroll";
import { LibraryAgentCard } from "../LibraryAgentCard/LibraryAgentCard";
import { LibraryFolder } from "../LibraryFolder/LibraryFolder";
import { LibrarySubSection } from "../LibrarySubSection/LibrarySubSection";
import { ArrowLeftIcon, HeartIcon } from "@phosphor-icons/react";
import { Text } from "@/components/atoms/Text/Text";
import {
  AnimatePresence,
  LayoutGroup,
  motion,
  useReducedMotion,
} from "framer-motion";
import { LibraryFolderEditDialog } from "../LibraryFolderEditDialog/LibraryFolderEditDialog";
import { LibraryFolderDeleteDialog } from "../LibraryFolderDeleteDialog/LibraryFolderDeleteDialog";
import type { LibraryTab, AgentStatusFilter, FleetSummary } from "../../types";
import { useLibraryAgentList } from "./useLibraryAgentList";
import { AgentBriefingPanel } from "../AgentBriefingPanel/AgentBriefingPanel";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useAgentStatusMap, getAgentStatus } from "../../hooks/useAgentStatus";

// cancels the current spring and starts a new one from current state.
const containerVariants = {
  hidden: {},
  show: {},
  exit: {
    opacity: 0,
    filter: "blur(4px)",
    transition: { duration: 0.12 },
  },
};

const reducedContainerVariants = {
  hidden: {},
  show: {},
  exit: {
    opacity: 0,
    transition: { duration: 0.12 },
  },
};

const itemInitial = {
  opacity: 0,
  filter: "blur(4px)",
};

const itemAnimate = {
  opacity: 1,
  filter: "blur(0px)",
};

const itemTransition = {
  type: "spring" as const,
  stiffness: 300,
  damping: 25,
  opacity: { duration: 0.2 },
  filter: { duration: 0.15 },
};

const reducedItemInitial = { opacity: 0 };
const reducedItemAnimate = { opacity: 1 };
const reducedItemTransition = { duration: 0.15 };

interface Props {
  searchTerm: string;
  librarySort: LibraryAgentSort;
  setLibrarySort: (value: LibraryAgentSort) => void;
  selectedFolderId: string | null;
  onFolderSelect: (folderId: string | null) => void;
  tabs: LibraryTab[];
  activeTab: string;
  onTabChange: (tabId: string) => void;
  statusFilter?: AgentStatusFilter;
  onStatusFilterChange?: (filter: AgentStatusFilter) => void;
  fleetSummary?: FleetSummary;
  briefingAgents?: LibraryAgent[];
}

export function LibraryAgentList({
  searchTerm,
  librarySort,
  setLibrarySort,
  selectedFolderId,
  onFolderSelect,
  tabs,
  activeTab,
  onTabChange,
  statusFilter = "all",
  onStatusFilterChange,
  fleetSummary,
  briefingAgents,
}: Props) {
  const isAgentBriefingEnabled = useGetFlag(Flag.AGENT_BRIEFING);
  const shouldReduceMotion = useReducedMotion();
  const activeContainerVariants = shouldReduceMotion
    ? reducedContainerVariants
    : containerVariants;
  const activeInitial = shouldReduceMotion ? reducedItemInitial : itemInitial;
  const activeAnimate = shouldReduceMotion ? reducedItemAnimate : itemAnimate;
  const activeTransition = shouldReduceMotion
    ? reducedItemTransition
    : itemTransition;

  const {
    isFavoritesTab,
    agentLoading,
    displayedCount,
    favoritesCount,
    agents,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
    foldersData,
    currentFolder,
    showFolders,
    editingFolder,
    setEditingFolder,
    deletingFolder,
    setDeletingFolder,
    handleAgentDrop,
    handleFolderDeleted,
  } = useLibraryAgentList({
    searchTerm,
    librarySort,
    selectedFolderId,
    onFolderSelect,
    activeTab,
    statusFilter,
  });

  const agentStatusMap = useAgentStatusMap(agents);

  return (
    <>
      {isAgentBriefingEnabled &&
        !selectedFolderId &&
        fleetSummary &&
        briefingAgents &&
        briefingAgents.length > 0 && (
          <div className="mb-4">
            <AgentBriefingPanel
              summary={fleetSummary}
              agents={briefingAgents}
            />
          </div>
        )}

      {!selectedFolderId && (
        <LibrarySubSection
          tabs={tabs}
          activeTab={activeTab}
          onTabChange={onTabChange}
          allCount={displayedCount}
          favoritesCount={favoritesCount}
          setLibrarySort={setLibrarySort}
          statusFilter={statusFilter}
          onStatusFilterChange={onStatusFilterChange}
          fleetSummary={fleetSummary}
        />
      )}

      <div className="pt-4">
        {selectedFolderId && (
          <div className="mb-4 flex items-center gap-2">
            <button
              type="button"
              onClick={() => onFolderSelect(null)}
              className="inline-flex items-center gap-1 text-sm text-zinc-500 hover:text-zinc-900"
            >
              <ArrowLeftIcon className="h-4 w-4" />
              My Library
            </button>
            {currentFolder && (
              <>
                <Text variant="body" className="text-zinc-400">
                  /
                </Text>
                <Text variant="large" className="text-zinc-700">
                  {currentFolder.name}
                </Text>
              </>
            )}
          </div>
        )}
        {agentLoading ? (
          <div className="flex h-[200px] items-center justify-center">
            <LoadingSpinner size="large" />
          </div>
        ) : isFavoritesTab && agents.length === 0 ? (
          <div className="flex h-[200px] flex-col items-center justify-center gap-2 text-zinc-500">
            <HeartIcon className="h-10 w-10" />
            <Text variant="body">No favorite agents yet</Text>
          </div>
        ) : (
          <InfiniteScroll
            isFetchingNextPage={isFetchingNextPage}
            fetchNextPage={fetchNextPage}
            hasNextPage={hasNextPage}
            loader={<LoadingSpinner size="medium" />}
          >
            <LayoutGroup>
              <AnimatePresence mode="popLayout">
                <motion.div
                  key={`${activeTab}-${selectedFolderId || "all"}`}
                  className="grid grid-cols-1 gap-6 sm:grid-cols-2 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
                  variants={activeContainerVariants}
                  initial="hidden"
                  animate="show"
                  exit="exit"
                >
                  {showFolders &&
                    foldersData?.folders.map((folder, i) => (
                      <motion.div
                        key={folder.id}
                        initial={activeInitial}
                        animate={activeAnimate}
                        transition={{
                          ...activeTransition,
                          delay: i * 0.04,
                        }}
                      >
                        <LibraryFolder
                          id={folder.id}
                          name={folder.name}
                          agentCount={folder.agent_count ?? 0}
                          color={folder.color ?? undefined}
                          icon={folder.icon ?? "📁"}
                          onAgentDrop={handleAgentDrop}
                          onClick={() => onFolderSelect(folder.id)}
                          onEdit={() => setEditingFolder(folder)}
                          onDelete={() => setDeletingFolder(folder)}
                        />
                      </motion.div>
                    ))}
                  {agents.map((agent, i) => (
                    <motion.div
                      key={agent.id}
                      initial={activeInitial}
                      animate={activeAnimate}
                      transition={{
                        ...activeTransition,
                        delay:
                          ((showFolders
                            ? (foldersData?.folders.length ?? 0)
                            : 0) +
                            i) *
                          0.04,
                      }}
                    >
                      <LibraryAgentCard
                        agent={agent}
                        statusInfo={getAgentStatus(
                          agentStatusMap,
                          agent.graph_id,
                        )}
                      />
                    </motion.div>
                  ))}
                </motion.div>
              </AnimatePresence>
            </LayoutGroup>
          </InfiniteScroll>
        )}
      </div>

      {editingFolder && (
        <LibraryFolderEditDialog
          folder={editingFolder}
          isOpen={!!editingFolder}
          setIsOpen={(open) => {
            if (!open) setEditingFolder(null);
          }}
        />
      )}

      {deletingFolder && (
        <LibraryFolderDeleteDialog
          folder={deletingFolder}
          isOpen={!!deletingFolder}
          setIsOpen={(open) => {
            if (!open) setDeletingFolder(null);
          }}
          onDeleted={handleFolderDeleted}
        />
      )}
    </>
  );
}
