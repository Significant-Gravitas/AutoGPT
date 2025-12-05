"use client";

import {
  getGetV1ListGraphExecutionsInfiniteQueryOptions,
  getV1GetGraphVersion,
  useDeleteV1DeleteGraphExecution,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import {
  getGetV2ListLibraryAgentsQueryKey,
  useDeleteV2DeleteLibraryAgent,
} from "@/app/api/__generated__/endpoints/library/library";
import {
  getGetV1ListExecutionSchedulesForAGraphQueryOptions,
  useDeleteV1DeleteExecutionSchedule,
} from "@/app/api/__generated__/endpoints/schedules/schedules";
import type { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
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
import { exportAsJSONFile } from "@/lib/utils";
import { DotsThreeIcon } from "@phosphor-icons/react";
import { useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

interface Props {
  agent: LibraryAgent;
  scheduleId?: string;
  run?: GraphExecution;
  agentGraphId?: string;
  onClearSelectedRun?: () => void;
}

export function AgentActionsDropdown({
  agent,
  run,
  agentGraphId,
  scheduleId,
  onClearSelectedRun,
}: Props) {
  const { toast } = useToast();

  const { mutateAsync: deleteAgent } = useDeleteV2DeleteLibraryAgent();

  const { mutateAsync: deleteRun, isPending: isDeletingRun } =
    useDeleteV1DeleteGraphExecution();

  const queryClient = useQueryClient();
  const router = useRouter();
  const [isDeletingAgent, setIsDeletingAgent] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [showDeleteRunDialog, setShowDeleteRunDialog] = useState(false);

  const { mutateAsync: deleteSchedule } = useDeleteV1DeleteExecutionSchedule();
  const [isDeletingSchedule, setIsDeletingSchedule] = useState(false);
  const [showDeleteScheduleDialog, setShowDeleteScheduleDialog] =
    useState(false);

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

  async function handleDeleteRun() {
    if (!run?.id || !agentGraphId) return;

    try {
      await deleteRun({ graphExecId: run.id });

      toast({ title: "Task deleted" });

      await queryClient.refetchQueries({
        queryKey:
          getGetV1ListGraphExecutionsInfiniteQueryOptions(agentGraphId)
            .queryKey,
      });

      if (onClearSelectedRun) onClearSelectedRun();

      setShowDeleteRunDialog(false);
    } catch (error: unknown) {
      toast({
        title: "Failed to delete task",
        description:
          error instanceof Error
            ? error.message
            : "An unexpected error occurred.",
        variant: "destructive",
      });
    }
  }

  async function handleDeleteSchedule() {
    setIsDeletingSchedule(true);
    try {
      await deleteSchedule({ scheduleId: scheduleId ?? "" });
      toast({ title: "Schedule deleted" });

      await queryClient.invalidateQueries({
        queryKey: getGetV1ListExecutionSchedulesForAGraphQueryOptions(
          agentGraphId ?? "",
        ).queryKey,
      });

      setShowDeleteDialog(false);
    } catch (error: unknown) {
      toast({
        title: "Failed to delete schedule",
        description:
          error instanceof Error
            ? error.message
            : "An unexpected error occurred.",
        variant: "destructive",
      });
    } finally {
      setIsDeletingSchedule(false);
    }
  }

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="icon"
            size="icon"
            aria-label="More actions"
            className="min-w-fit"
          >
            <DotsThreeIcon size={18} />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          {run ? (
            <>
              <DropdownMenuItem
                onClick={() => setShowDeleteRunDialog(true)}
                className="flex items-center gap-2"
              >
                Delete this task
              </DropdownMenuItem>
              <DropdownMenuSeparator />
            </>
          ) : null}
          <DropdownMenuItem asChild>
            <Link
              href={`/build?flowID=${agent.graph_id}&flowVersion=${agent.graph_version}`}
              target="_blank"
              className="flex items-center gap-2"
            >
              Edit agent
            </Link>
          </DropdownMenuItem>
          <DropdownMenuItem
            onClick={handleExport}
            className="flex items-center gap-2"
          >
            Export agent to file
          </DropdownMenuItem>
          <DropdownMenuItem
            onClick={() => setShowDeleteDialog(true)}
            className="flex items-center gap-2"
          >
            Delete agent
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <Dialog
        controlled={{
          isOpen: showDeleteRunDialog,
          set: setShowDeleteRunDialog,
        }}
        styling={{ maxWidth: "32rem" }}
        title="Delete task"
      >
        <Dialog.Content>
          <div>
            <Text variant="large">
              Are you sure you want to delete this task? This action cannot be
              undone.
            </Text>
            <Dialog.Footer>
              <Button
                variant="secondary"
                disabled={isDeletingRun}
                onClick={() => setShowDeleteRunDialog(false)}
              >
                Cancel
              </Button>
              <Button
                variant="destructive"
                onClick={handleDeleteRun}
                loading={isDeletingRun}
              >
                Delete Task
              </Button>
            </Dialog.Footer>
          </div>
        </Dialog.Content>
      </Dialog>

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

      <Dialog
        controlled={{
          isOpen: showDeleteScheduleDialog,
          set: setShowDeleteScheduleDialog,
        }}
        styling={{ maxWidth: "32rem" }}
        title="Delete schedule"
      >
        <Dialog.Content>
          <div>
            <Text variant="large">
              Are you sure you want to delete this schedule? This action cannot
              be undone.
            </Text>
            <Dialog.Footer>
              <Button
                variant="secondary"
                disabled={isDeletingSchedule}
                onClick={() => setShowDeleteScheduleDialog(false)}
              >
                Cancel
              </Button>
              <Button
                variant="destructive"
                onClick={handleDeleteSchedule}
                loading={isDeletingSchedule}
              >
                Delete Schedule
              </Button>
            </Dialog.Footer>
          </div>
        </Dialog.Content>
      </Dialog>
    </>
  );
}
