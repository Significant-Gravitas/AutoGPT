"use client";

import {
  getGetV2ListLibraryAgentsQueryKey,
  useDeleteV2DeleteLibraryAgent,
  usePostV2ForkLibraryAgent,
} from "@/app/api/__generated__/endpoints/library/library";
import {
  usePostV2BulkMoveAgents,
  getGetV2ListLibraryFoldersQueryKey,
} from "@/app/api/__generated__/endpoints/folders/folders";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { DotsThree } from "@phosphor-icons/react";
import { useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { MoveToFolderDialog } from "../../MoveToFolderDialog/MoveToFolderDialog";

interface AgentCardMenuProps {
  agent: LibraryAgent;
}

export function AgentCardMenu({ agent }: AgentCardMenuProps) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const router = useRouter();
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [showMoveDialog, setShowMoveDialog] = useState(false);
  const [isDeletingAgent, setIsDeletingAgent] = useState(false);
  const [isDuplicatingAgent, setIsDuplicatingAgent] = useState(false);
  const [isRemovingFromFolder, setIsRemovingFromFolder] = useState(false);

  const { mutateAsync: deleteAgent } = useDeleteV2DeleteLibraryAgent();
  const { mutateAsync: forkAgent } = usePostV2ForkLibraryAgent();
  const { mutateAsync: bulkMoveAgents } = usePostV2BulkMoveAgents({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({
          queryKey: getGetV2ListLibraryAgentsQueryKey(),
        });
        queryClient.invalidateQueries({
          queryKey: getGetV2ListLibraryFoldersQueryKey(),
        });
      },
    },
  });

  async function handleDuplicateAgent() {
    if (!agent.id) return;

    setIsDuplicatingAgent(true);

    try {
      const result = await forkAgent({ libraryAgentId: agent.id });

      if (result.status === 200) {
        await queryClient.refetchQueries({
          queryKey: getGetV2ListLibraryAgentsQueryKey(),
        });

        toast({
          title: "Agent duplicated",
          description: `${result.data.name} has been created.`,
        });
      }
    } catch (error: unknown) {
      toast({
        title: "Failed to duplicate agent",
        description:
          error instanceof Error
            ? error.message
            : "An unexpected error occurred.",
        variant: "destructive",
      });
    } finally {
      setIsDuplicatingAgent(false);
    }
  }

  async function handleRemoveFromFolder() {
    if (!agent.id) return;

    setIsRemovingFromFolder(true);

    try {
      await bulkMoveAgents({
        data: {
          agent_ids: [agent.id],
          folder_id: null,
        },
      });

      toast({
        title: "Removed from folder",
        description: "Agent has been moved back to your library.",
      });
    } catch (error: unknown) {
      toast({
        title: "Failed to remove from folder",
        description:
          error instanceof Error
            ? error.message
            : "An unexpected error occurred.",
        variant: "destructive",
      });
    } finally {
      setIsRemovingFromFolder(false);
    }
  }

  async function handleDeleteAgent() {
    if (!agent.id) return;

    setIsDeletingAgent(true);

    try {
      await deleteAgent({ libraryAgentId: agent.id });

      await queryClient.refetchQueries({
        queryKey: getGetV2ListLibraryAgentsQueryKey(),
      });

      toast({ title: "Agent deleted" });
      setShowDeleteDialog(false);
      router.push("/library");
    } catch (error: unknown) {
      toast({
        title: "Failed to delete agent",
        description:
          error instanceof Error
            ? error.message
            : "An unexpected error occurred.",
        variant: "destructive",
      });
    } finally {
      setIsDeletingAgent(false);
    }
  }

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button
            className="absolute right-2 top-1 rounded p-1.5 transition-opacity hover:bg-neutral-100"
            onClick={(e) => e.stopPropagation()}
            aria-label="More actions"
          >
            <DotsThree className="h-5 w-5 text-neutral-600" />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          {agent.can_access_graph && (
            <>
              <DropdownMenuItem asChild>
                <Link
                  href={`/build?flowID=${agent.graph_id}&flowVersion=${agent.graph_version}`}
                  target="_blank"
                  className="flex items-center gap-2"
                  onClick={(e) => e.stopPropagation()}
                >
                  Edit agent
                </Link>
              </DropdownMenuItem>
              <DropdownMenuSeparator />
            </>
          )}
          <DropdownMenuItem
            onClick={(e) => {
              e.stopPropagation();
              handleDuplicateAgent();
            }}
            disabled={isDuplicatingAgent}
            className="flex items-center gap-2"
          >
            Duplicate agent
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem
            onClick={(e) => {
              e.stopPropagation();
              setShowMoveDialog(true);
            }}
            className="flex items-center gap-2"
          >
            Move to folder
          </DropdownMenuItem>
          {agent.folder_id && (
            <>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                onClick={(e) => {
                  e.stopPropagation();
                  handleRemoveFromFolder();
                }}
                disabled={isRemovingFromFolder}
                className="flex items-center gap-2"
              >
                Remove from folder
              </DropdownMenuItem>
            </>
          )}
          <DropdownMenuSeparator />
          <DropdownMenuItem
            onClick={(e) => {
              e.stopPropagation();
              setShowDeleteDialog(true);
            }}
            className="flex items-center gap-2 text-red-600 focus:bg-red-50 focus:text-red-600"
          >
            Delete agent
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <Dialog
        controlled={{
          isOpen: showDeleteDialog,
          set: setShowDeleteDialog,
        }}
        styling={{ maxWidth: "32rem" }}
        title="Delete agent"
      >
        <Dialog.Content>
          <div>
            <Text variant="large">
              Are you sure you want to delete this agent? This action cannot be
              undone.
            </Text>
            <Dialog.Footer>
              <Button
                variant="secondary"
                disabled={isDeletingAgent}
                onClick={() => setShowDeleteDialog(false)}
              >
                Cancel
              </Button>
              <Button
                variant="destructive"
                onClick={handleDeleteAgent}
                loading={isDeletingAgent}
              >
                Delete Agent
              </Button>
            </Dialog.Footer>
          </div>
        </Dialog.Content>
      </Dialog>

      <MoveToFolderDialog
        agentId={agent.id}
        agentName={agent.name}
        currentFolderId={agent.folder_id}
        isOpen={showMoveDialog}
        setIsOpen={setShowMoveDialog}
      />
    </>
  );
}
