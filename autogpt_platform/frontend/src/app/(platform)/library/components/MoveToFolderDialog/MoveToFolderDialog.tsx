"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useToast } from "@/components/molecules/Toast/use-toast";
import {
  useGetV2ListLibraryFolders,
  usePostV2BulkMoveAgents,
  getGetV2ListLibraryFoldersQueryKey,
} from "@/app/api/__generated__/endpoints/folders/folders";
import { getGetV2ListLibraryAgentsQueryKey } from "@/app/api/__generated__/endpoints/library/library";
import { okData } from "@/app/api/helpers";
import { useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import type { LibraryFolder } from "@/app/api/__generated__/models/libraryFolder";

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

  const { data: foldersData } = useGetV2ListLibraryFolders(undefined, {
    query: { select: okData },
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
          title: "Agent moved",
          description: `"${agentName}" has been moved.`,
        });
      },
      onError: () => {
        toast({
          title: "Error",
          description: "Failed to move agent. Please try again.",
          variant: "destructive",
        });
      },
    },
  });

  const hierarchicalFolders = useMemo(() => {
    const allFolders = foldersData?.folders ?? [];
    const result: { folder: LibraryFolder; depth: number }[] = [];

    function addChildren(parentId: string | null, depth: number) {
      for (const f of allFolders) {
        if ((f.parent_id ?? null) === parentId) {
          result.push({ folder: f, depth });
          addChildren(f.id, depth + 1);
        }
      }
    }

    addChildren(null, 0);
    return result;
  }, [foldersData]);

  const folders = hierarchicalFolders.filter(
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
          <div className="max-h-[280px] overflow-y-auto">
            {folders.length === 0 ? (
              <div className="flex h-20 items-center justify-center">
                <Text variant="small" className="text-zinc-400">
                  No folders found
                </Text>
              </div>
            ) : (
              <div className="flex flex-col gap-1">
                {currentFolderId && (
                  <Button
                    variant="ghost"
                    className="w-full justify-start gap-3 px-3 py-2.5"
                    disabled={isPending}
                    onClick={() => handleMoveToFolder(null)}
                  >
                    <span className="text-lg">📂</span>
                    <div className="flex flex-col items-start">
                      <Text variant="small-medium">My Library (root)</Text>
                      <Text variant="small" className="text-zinc-400">
                        Remove from folder
                      </Text>
                    </div>
                  </Button>
                )}
                {folders.map(({ folder, depth }) => (
                  <Button
                    key={folder.id}
                    variant="ghost"
                    className="w-full justify-start gap-3 py-2.5"
                    style={{ paddingLeft: `${0.75 + depth * 1.25}rem` }}
                    disabled={isPending}
                    onClick={() => handleMoveToFolder(folder.id)}
                  >
                    <span className="text-lg">{folder.icon ?? "📁"}</span>
                    <div className="flex flex-col items-start">
                      <Text variant="small-medium">{folder.name}</Text>
                      <Text variant="small" className="text-zinc-400">
                        {folder.agent_count ?? 0}{" "}
                        {(folder.agent_count ?? 0) === 1 ? "agent" : "agents"}
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
