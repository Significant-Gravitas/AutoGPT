"use client";

import {
  getGetV1ListGraphExecutionsQueryKey,
  usePostV1ExecuteGraphAgent,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import {
  getGetV1ListExecutionSchedulesForAGraphQueryOptions,
  useDeleteV1DeleteExecutionSchedule,
} from "@/app/api/__generated__/endpoints/schedules/schedules";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { okData } from "@/app/api/helpers";
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
import { DotsThreeVertical } from "@phosphor-icons/react";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

interface Props {
  agent: LibraryAgent;
  schedule: GraphExecutionJobInfo;
  onDeleted?: () => void;
  onRunCreated?: (runID: string) => void;
}

export function ScheduleActionsDropdown({
  agent,
  schedule,
  onDeleted,
  onRunCreated,
}: Props) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);

  const { mutateAsync: deleteSchedule, isPending: isDeleting } =
    useDeleteV1DeleteExecutionSchedule();

  const { mutateAsync: executeAgent, isPending: isRunning } =
    usePostV1ExecuteGraphAgent();

  async function handleDelete() {
    try {
      await deleteSchedule({ scheduleId: schedule.id });

      toast({ title: "Schedule deleted" });
      setShowDeleteDialog(false);

      onDeleted?.();

      queryClient.invalidateQueries({
        queryKey: getGetV1ListExecutionSchedulesForAGraphQueryOptions(
          agent.graph_id,
        ).queryKey,
      });
    } catch (error: unknown) {
      toast({
        title: "Failed to delete schedule",
        description:
          error instanceof Error
            ? error.message
            : "An unexpected error occurred.",
        variant: "destructive",
      });
    }
  }

  async function handleRunNow(e: React.MouseEvent) {
    e.stopPropagation();

    try {
      toast({ title: "Run started" });

      const res = await executeAgent({
        graphId: schedule.graph_id,
        graphVersion: schedule.graph_version,
        data: {
          inputs: schedule.input_data || {},
          credentials_inputs: schedule.input_credentials || {},
          source: "library",
        },
      });

      const newRunID = okData(res)?.id;

      await queryClient.invalidateQueries({
        queryKey: getGetV1ListGraphExecutionsQueryKey(agent.graph_id),
      });

      if (newRunID) {
        onRunCreated?.(newRunID);
      }
    } catch (error: unknown) {
      toast({
        title: "Failed to start run",
        description:
          error instanceof Error
            ? error.message
            : "An unexpected error occurred.",
        variant: "destructive",
      });
    }
  }

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button
            className="ml-auto shrink-0 rounded p-1 hover:bg-gray-100"
            onClick={(e) => e.stopPropagation()}
            aria-label="More actions"
          >
            <DotsThreeVertical className="h-5 w-5 text-gray-400" />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem
            onClick={handleRunNow}
            disabled={isRunning}
            className="flex items-center gap-2"
          >
            {isRunning ? "Running..." : "Run now"}
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem
            onClick={(e) => {
              e.stopPropagation();
              setShowDeleteDialog(true);
            }}
            className="flex items-center gap-2"
          >
            Delete schedule
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <Dialog
        controlled={{
          isOpen: showDeleteDialog,
          set: setShowDeleteDialog,
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
                Delete Schedule
              </Button>
            </Dialog.Footer>
          </div>
        </Dialog.Content>
      </Dialog>
    </>
  );
}
