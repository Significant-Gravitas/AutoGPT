"use client";

import { LibraryAgentSort } from "@/app/api/__generated__/models/libraryAgentSort";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { InfiniteScroll } from "@/components/contextual/InfiniteScroll/InfiniteScroll";
import { LibraryActionSubHeader } from "../LibraryActionSubHeader/LibraryActionSubHeader";
import { LibraryAgentCard } from "../LibraryAgentCard/LibraryAgentCard";
import { LibraryFolder } from "../LibraryFolder/LibraryFolder";
import { LibrarySubSection } from "../LibrarySubSection/LibrarySubSection";
import { Button } from "@/components/atoms/Button/Button";
import { ArrowLeftIcon, HeartIcon } from "@phosphor-icons/react";
import { Text } from "@/components/atoms/Text/Text";
import { Tab } from "../LibraryTabs/LibraryTabs";
import { LayoutGroup } from "framer-motion";
import { LibraryFolderEditDialog } from "../LibraryFolderEditDialog/LibraryFolderEditDialog";
import { LibraryFolderDeleteDialog } from "../LibraryFolderDeleteDialog/LibraryFolderDeleteDialog";
import { useLibraryAgentList } from "./useLibraryAgentList";

interface Props {
  searchTerm: string;
  librarySort: LibraryAgentSort;
  setLibrarySort: (value: LibraryAgentSort) => void;
  selectedFolderId: string | null;
  onFolderSelect: (folderId: string | null) => void;
  tabs: Tab[];
  activeTab: string;
  onTabChange: (tabId: string) => void;
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
}: Props) {
  const {
    isFavoritesTab,
    agentLoading,
    agentCount,
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
  });

  return (
    <>
      <LibraryActionSubHeader
        agentCount={agentCount}
        setLibrarySort={setLibrarySort}
      />
      {!selectedFolderId && (
        <LibrarySubSection
          tabs={tabs}
          activeTab={activeTab}
          onTabChange={onTabChange}
        />
      )}

      <div>
        {selectedFolderId && (
          <div className="mb-4 flex items-center gap-3">
            <Button
              variant="ghost"
              size="small"
              onClick={() => onFolderSelect(null)}
              className="gap-2"
            >
              <ArrowLeftIcon className="h-4 w-4" />
              Back to Library
            </Button>
            {currentFolder && (
              <Text variant="h4" className="text-zinc-700">
                {currentFolder.icon} {currentFolder.name}
              </Text>
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
              <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                {showFolders &&
                  foldersData?.folders.map((folder) => (
                    <LibraryFolder
                      key={folder.id}
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
                  ))}
                {agents.map((agent) => (
                  <LibraryAgentCard key={agent.id} agent={agent} />
                ))}
              </div>
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
