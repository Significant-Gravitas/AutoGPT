"use client";

import React, { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import Link from "next/link";
import {
  FileArrowDownIcon,
  PencilSimpleIcon,
  TrashIcon,
} from "@phosphor-icons/react";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { getV1GetGraphVersion } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { exportAsJSONFile } from "@/lib/utils";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useRouter } from "next/navigation";
import { useDeleteV2DeleteLibraryAgent } from "@/app/api/__generated__/endpoints/library/library";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  agent: LibraryAgent;
}

export function AgentActionsDropdown({ agent }: Props) {
  const { toast } = useToast();
  const { mutateAsync: deleteAgent } = useDeleteV2DeleteLibraryAgent();
  const router = useRouter();
  const [isDeleting, setIsDeleting] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);

  async function handleDelete() {
    if (!agent.id) return;

    setIsDeleting(true);

    try {
      await deleteAgent({ libraryAgentId: agent.id });
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
      setIsDeleting(false);
    }
  }

  async function handleExport() {
    try {
      const res = await getV1GetGraphVersion(
        agent.graph_id,
        agent.graph_version,
        { for_export: true },
      );
      if (res.status === 200) {
        const filename = `${agent.name}_v${agent.graph_version}.json`;
        exportAsJSONFile(res.data as any, filename);
        toast({ title: "Agent exported" });
      } else {
        toast({ title: "Failed to export agent", variant: "destructive" });
      }
    } catch (e: any) {
      toast({
        title: "Failed to export agent",
        description: e?.message,
        variant: "destructive",
      });
    }
  }

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="secondary" size="small" className="min-w-fit">
            •••
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem asChild>
            <Link
              href={`/build?flowID=${agent.graph_id}&flowVersion=${agent.graph_version}`}
              target="_blank"
              className="flex items-center gap-2"
            >
              <PencilSimpleIcon size={16} /> Edit agent
            </Link>
          </DropdownMenuItem>
          <DropdownMenuItem
            onClick={handleExport}
            className="flex items-center gap-2"
          >
            <FileArrowDownIcon size={16} /> Export agent
          </DropdownMenuItem>
          <DropdownMenuItem
            onClick={() => setShowDeleteDialog(true)}
            className="flex items-center gap-2"
          >
            <TrashIcon size={16} /> Delete agent
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
                disabled={isDeleting}
                onClick={() => setShowDeleteDialog(false)}
              >
                Cancel
              </Button>
              <Button
                variant="destructive"
                onClick={handleDelete}
                loading={isDeleting}
              >
                Delete
              </Button>
            </Dialog.Footer>
          </div>
        </Dialog.Content>
      </Dialog>
    </>
  );
}
