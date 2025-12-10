"use client";

import {
  getGetV2ListPresetsQueryKey,
  useDeleteV2DeleteAPreset,
} from "@/app/api/__generated__/endpoints/presets/presets";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import type { LibraryAgentPresetResponse } from "@/app/api/__generated__/models/libraryAgentPresetResponse";
import { okData } from "@/app/api/helpers";
import { Button } from "@/components/atoms/Button/Button";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { FloppyDiskIcon, TrashIcon } from "@phosphor-icons/react";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { AgentActionsDropdown } from "../../AgentActionsDropdown";

interface Props {
  agent: LibraryAgent;
  triggerId: string;
  onDeleted?: () => void;
  onSaveChanges?: () => void;
  isSaving?: boolean;
  onSwitchToRunsTab?: () => void;
}

export function SelectedTriggerActions({
  agent,
  triggerId,
  onDeleted,
  onSaveChanges,
  isSaving,
  onSwitchToRunsTab,
}: Props) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);

  const deleteMutation = useDeleteV2DeleteAPreset({
    mutation: {
      onSuccess: async () => {
        toast({
          title: "Trigger deleted",
        });
        const queryKey = getGetV2ListPresetsQueryKey({
          graph_id: agent.graph_id,
        });

        queryClient.invalidateQueries({
          queryKey,
        });

        const queryData = queryClient.getQueryData<{
          data: LibraryAgentPresetResponse;
        }>(queryKey);

        const presets =
          okData<LibraryAgentPresetResponse>(queryData)?.presets ?? [];
        const triggers = presets.filter(
          (preset) => preset.webhook_id && preset.webhook,
        );

        setShowDeleteDialog(false);
        onDeleted?.();

        if (triggers.length === 0 && onSwitchToRunsTab) {
          onSwitchToRunsTab();
        }
      },
      onError: (error: any) => {
        toast({
          title: "Failed to delete trigger",
          description: error.message || "An unexpected error occurred.",
          variant: "destructive",
        });
      },
    },
  });

  function handleDelete() {
    deleteMutation.mutate({ presetId: triggerId });
  }

  return (
    <>
      <div className="my-4 flex flex-col items-center gap-3">
        <Button
          variant="icon"
          size="icon"
          aria-label="Save changes"
          onClick={onSaveChanges}
          disabled={isSaving || deleteMutation.isPending}
        >
          {isSaving ? (
            <LoadingSpinner size="small" />
          ) : (
            <FloppyDiskIcon weight="bold" size={18} className="text-zinc-700" />
          )}
        </Button>
        <Button
          variant="icon"
          size="icon"
          aria-label="Delete trigger"
          onClick={() => setShowDeleteDialog(true)}
          disabled={isSaving || deleteMutation.isPending}
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
        title="Delete trigger"
      >
        <Dialog.Content>
          <Text variant="large">
            Are you sure you want to delete this trigger? This action cannot be
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
