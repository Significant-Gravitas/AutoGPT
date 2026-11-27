"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useToast } from "@/components/molecules/Toast/use-toast";
import {
  useGetV2GetFolderTree,
  usePostV2BulkMoveAgents,
  getGetV2ListLibraryFoldersQueryKey,
} from "@/app/api/__generated__/endpoints/folders/folders";
import { getGetV2ListLibraryAgentsQueryKey } from "@/app/api/__generated__/endpoints/library/library";
import { okData } from "@/app/api/helpers";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { flattenFolderTree } from "./helpers";

interface Props {
  agentId: string;
  agentName: string;
  currentFolderId?: string | null;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
}

export function MoveToFolderDialog({
  agentId,
  agentName,
  currentFolderId,
  isOpen,
  setIsOpen,
}: Props) {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [search, setSearch] = useState("");

  // The whole tree, subfolders included, read fresh on every open: folder
  // changes (and Otto's) don't invalidate the tree's query key.
  const { data: folderTree, isLoading: isLoadingFolders } =
    useGetV2GetFolderTree({
      query: { select: okData, enabled: isOpen, staleTime: 0 },
    });

  const { mutate: moveAgent, isPending } = usePostV2BulkMoveAgents({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({
          queryKey: getGetV2ListLibraryAgentsQueryKey(),
        });
        queryClient.invalidateQueries({
          queryKey: getGetV2ListLibraryFoldersQueryKey(),
        });
        setIsOpen(false);
        setSearch("");
        toast({
          title: "Workflow moved",
          description: `"${agentName}" has been moved.`,
        });
      },
      onError: () => {
        toast({
          title: "Error",
          description: "Failed to move workflow. Please try again.",
          variant: "destructive",
        });
      },
    },
  });

  const folders = flattenFolderTree(folderTree?.tree ?? []).filter(
    ({ folder: f }) =>
      f.id !== currentFolderId &&
      f.name.toLowerCase().includes(search.toLowerCase()),
  );

  function handleMoveToFolder(folderId: string | null) {
    moveAgent({
      data: {
        agent_ids: [agentId],
        folder_id: folderId,
      },
    });
  }

  return (
    <Dialog
      controlled={{ isOpen, set: setIsOpen }}
      styling={{ maxWidth: "28rem" }}
      title="Move to folder"
      onClose={() => {
        setSearch("");
      }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-3">
          <Input
            id="search-folders"
            label="Search folders"
            placeholder="Search folders..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full"
          />
          <div className="flex max-h-[280px] flex-col gap-1 overflow-y-auto">
            {currentFolderId && (
              <Button
                variant="ghost"
                className="w-full justify-start"
                disabled={isPending}
                onClick={() => handleMoveToFolder(null)}
              >
                <span className="text-lead">📂</span>
                <div className="flex flex-col items-start">
                  <Text variant="small" weight="medium">
                    My Library (root)
                  </Text>
                  <Text variant="small" tone="subtle">
                    Remove from folder
                  </Text>
                </div>
              </Button>
            )}
            {folders.length === 0 ? (
              <div className="flex h-20 items-center justify-center">
                <Text variant="small" tone="subtle">
                  {isLoadingFolders ? "Loading folders…" : "No folders found"}
                </Text>
              </div>
            ) : (
              <div className="flex flex-col gap-1">
                {folders.map(({ folder, depth }) => (
                  <Button
                    key={folder.id}
                    variant="ghost"
                    className="w-full justify-start"
                    style={{ "--folder-depth": depth } as React.CSSProperties}
                    disabled={isPending}
                    onClick={() => handleMoveToFolder(folder.id)}
                  >
                    <span className="ml-[calc(var(--folder-depth)*1.25rem)] text-lead">
                      {folder.icon ?? "📁"}
                    </span>
                    <div className="flex flex-col items-start">
                      <Text variant="small" weight="medium">
                        {folder.name}
                      </Text>
                      <Text variant="small" tone="subtle">
                        {folder.agent_count ?? 0}{" "}
                        {(folder.agent_count ?? 0) === 1
                          ? "workflow"
                          : "workflows"}
                      </Text>
                    </div>
                  </Button>
                ))}
              </div>
            )}
          </div>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
