import { useScheduleDetailHeader } from "../../RunDetailHeader/useScheduleDetailHeader";
import { useDeleteV1DeleteExecutionSchedule } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useState } from "react";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  ArrowSquareOut,
  PencilSimpleIcon,
  TrashIcon,
} from "@phosphor-icons/react";

type Props = {
  agent: LibraryAgent;
  scheduleId: string;
  onDeleted?: () => void;
};

export function ScheduleActions({ agent, scheduleId, onDeleted }: Props) {
  const { toast } = useToast();
  const { mutateAsync: deleteSchedule } = useDeleteV1DeleteExecutionSchedule();
  const [isDeleting, setIsDeleting] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);

  const { openInBuilderHref } = useScheduleDetailHeader(
    agent.graph_id,
    scheduleId,
    agent.graph_version,
  );

  async function handleDelete() {
    setIsDeleting(true);
    try {
      await deleteSchedule({ scheduleId });
      toast({ title: "Schedule deleted" });
      setShowDeleteDialog(false);
      if (onDeleted) onDeleted();
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
      setIsDeleting(false);
    }
  }

  return (
    <>
      <div className="flex items-center gap-2">
        {openInBuilderHref && (
          <Button
            variant="secondary"
            size="small"
            as="NextLink"
            href={openInBuilderHref}
          >
            <ArrowSquareOut size={14} /> Open in builder
          </Button>
        )}
        <Button
          variant="secondary"
          size="small"
          as="NextLink"
          href={`/build?flowID=${agent.graph_id}&flowVersion=${agent.graph_version}`}
        >
          <PencilSimpleIcon size={16} /> Edit agent
        </Button>
        <Button
          variant="secondary"
          size="small"
          onClick={() => setShowDeleteDialog(true)}
        >
          <TrashIcon size={16} /> Delete
        </Button>
      </div>

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
                Delete
              </Button>
            </Dialog.Footer>
          </div>
        </Dialog.Content>
      </Dialog>
    </>
  );
}
