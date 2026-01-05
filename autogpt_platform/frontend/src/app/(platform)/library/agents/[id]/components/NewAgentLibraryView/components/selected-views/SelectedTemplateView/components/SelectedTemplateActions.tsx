"use client";

import {
  getGetV2ListPresetsQueryKey,
  getV2ListPresets,
  useDeleteV2DeleteAPreset,
} from "@/app/api/__generated__/endpoints/presets/presets";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { okData } from "@/app/api/helpers";
import { Button } from "@/components/atoms/Button/Button";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { FloppyDiskIcon, PlayIcon, TrashIcon } from "@phosphor-icons/react";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { AgentActionsDropdown } from "../../AgentActionsDropdown";

interface Props {
  agent: LibraryAgent;
  templateId: string;
  onDeleted?: () => void;
  onSaveChanges?: () => void;
  onStartTask?: () => void;
  isSaving?: boolean;
  isStarting?: boolean;
  onSwitchToRunsTab?: () => void;
}

export function SelectedTemplateActions({
  agent,
  templateId,
  onDeleted,
  onSaveChanges,
  onStartTask,
  isSaving,
  isStarting,
  onSwitchToRunsTab,
}: Props) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);

  const deleteMutation = useDeleteV2DeleteAPreset({
    mutation: {
      onSuccess: async () => {
        toast({
          title: "Template deleted",
        });
        const queryKey = getGetV2ListPresetsQueryKey({
          graph_id: agent.graph_id,
        });

        queryClient.invalidateQueries({
          queryKey,
        });

        const queryData =
          queryClient.getQueryData<
            Awaited<ReturnType<typeof getV2ListPresets>>
          >(queryKey);

        const presets = okData(queryData)?.presets ?? [];
        const templates = presets.filter((preset) => !preset.webhook_id);

        setShowDeleteDialog(false);
        onDeleted?.();

        if (templates.length === 0 && onSwitchToRunsTab) {
          onSwitchToRunsTab();
        }
      },
      onError: (error: any) => {
        toast({
          title: "Failed to delete template",
          description: error.message || "An unexpected error occurred.",
          variant: "destructive",
        });
      },
    },
  });

  function handleDelete() {
    deleteMutation.mutate({ presetId: templateId });
  }

  return (
    <>
      <div className="my-4 flex flex-col items-center gap-3">
        <Button
          variant="icon"
          size="icon"
          aria-label="Save changes"
          onClick={onSaveChanges}
          disabled={isSaving || isStarting || deleteMutation.isPending}
        >
          {isSaving ? (
            <LoadingSpinner size="small" />
          ) : (
            <FloppyDiskIcon weight="bold" size={18} className="text-zinc-700" />
          )}
        </Button>
        {onStartTask && (
          <Button
            variant="icon"
            size="icon"
            aria-label="Start task from template"
            onClick={onStartTask}
            disabled={isSaving || isStarting || deleteMutation.isPending}
          >
            {isStarting ? (
              <>
                <LoadingSpinner size="small" />
              </>
            ) : (
              <>
                <PlayIcon weight="bold" size={16} />
              </>
            )}
          </Button>
        )}
        <Button
          variant="icon"
          size="icon"
          aria-label="Delete template"
          onClick={() => setShowDeleteDialog(true)}
          disabled={isSaving || isStarting || deleteMutation.isPending}
        >
          {deleteMutation.isPending ? (
            <LoadingSpinner size="small" />
          ) : (
            <TrashIcon weight="bold" size={18} />
          )}
        </Button>
        <AgentActionsDropdown agent={agent} />
      </div>

      <Dialog
        controlled={{
          isOpen: showDeleteDialog,
          set: setShowDeleteDialog,
        }}
        styling={{ maxWidth: "32rem" }}
        title="Delete template"
      >
        <Dialog.Content>
          <Text variant="large">
            Are you sure you want to delete this template? This action cannot be
            undone.
          </Text>
          <Dialog.Footer>
            <Button
              variant="secondary"
              onClick={() => setShowDeleteDialog(false)}
              disabled={deleteMutation.isPending}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleDelete}
              loading={deleteMutation.isPending}
            >
              Delete
            </Button>
          </Dialog.Footer>
        </Dialog.Content>
      </Dialog>
    </>
  );
}
