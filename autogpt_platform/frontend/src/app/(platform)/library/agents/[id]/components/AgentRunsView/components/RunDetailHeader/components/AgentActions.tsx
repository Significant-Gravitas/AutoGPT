"use client";

import React, { useCallback } from "react";
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
import { useRouter } from "next/navigation";
import { useDeleteV2DeleteLibraryAgent } from "@/app/api/__generated__/endpoints/library/library";
import { getV1GetGraphVersion } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { exportAsJSONFile } from "@/lib/utils";
import { useToast } from "@/components/molecules/Toast/use-toast";

interface AgentActionsProps {
  agent: LibraryAgent;
}

export function AgentActions({ agent }: AgentActionsProps) {
  const router = useRouter();
  const { toast } = useToast();
  const deleteMutation = useDeleteV2DeleteLibraryAgent();

  const handleExport = useCallback(async () => {
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
  }, [agent.graph_id, agent.graph_version, agent.name, toast]);

  const handleDelete = useCallback(() => {
    if (!agent?.id) return;
    const confirmed = window.confirm(
      "Are you sure you want to delete this agent? This action cannot be undone.",
    );
    if (!confirmed) return;
    deleteMutation.mutate(
      { libraryAgentId: agent.id },
      {
        onSuccess: () => {
          toast({ title: "Agent deleted" });
          router.push("/library");
        },
        onError: (error: any) =>
          toast({
            title: "Failed to delete agent",
            description: error?.message,
            variant: "destructive",
          }),
      },
    );
  }, [agent.id, deleteMutation, router, toast]);

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="secondary" size="small">
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
          <FileArrowDownIcon size={16} /> Export agent to file
        </DropdownMenuItem>
        <DropdownMenuItem
          onClick={handleDelete}
          className="flex items-center gap-2"
        >
          <TrashIcon size={16} /> Delete agent
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
