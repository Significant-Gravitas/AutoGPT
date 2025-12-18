"use client";

import {
  getGetV1ListExecutionSchedulesForAGraphQueryOptions,
  useDeleteV1DeleteExecutionSchedule,
} from "@/app/api/__generated__/endpoints/schedules/schedules";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
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
}

export function ScheduleActionsDropdown({ agent, schedule, onDeleted }: Props) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);

  const { mutateAsync: deleteSchedule, isPending: isDeleting } =
    useDeleteV1DeleteExecutionSchedule();

  async function handleDelete() {
    try {
      await deleteSchedule({ scheduleId: schedule.id });

      toast({ title: "Schedule deleted" });

      queryClient.invalidateQueries({
        queryKey: getGetV1ListExecutionSchedulesForAGraphQueryOptions(
          agent.graph_id,
        ).queryKey,
      });

      setShowDeleteDialog(false);
      onDeleted?.();
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
