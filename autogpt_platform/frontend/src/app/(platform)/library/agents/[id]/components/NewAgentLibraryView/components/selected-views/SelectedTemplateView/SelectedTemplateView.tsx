"use client";

import React, { useCallback, useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import {
  PencilIcon,
  PlayIcon,
  StopIcon,
  TrashIcon,
} from "@phosphor-icons/react";
import {
  TabsLine,
  TabsLineContent,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";
import { useToast } from "@/components/molecules/Toast/use-toast";
import {
  useDeleteV2DeleteAPreset,
  getGetV2ListPresetsQueryKey,
  useGetV2GetASpecificPreset,
  getGetV2GetASpecificPresetQueryKey,
  usePatchV2UpdateAnExistingPreset,
} from "@/app/api/__generated__/endpoints/presets/presets";
import { getGetV1ListGraphExecutionsQueryKey } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { getGetV1ListExecutionSchedulesForAGraphQueryKey } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { RunDetailCard } from "../RunDetailCard/RunDetailCard";
import { Skeleton } from "@/components/__legacy__/ui/skeleton";
import { AgentInputsReadOnly } from "../../modals/AgentInputsReadOnly/AgentInputsReadOnly";
import { RunAgentModal } from "../../modals/RunAgentModal/RunAgentModal";
import { okData } from "@/app/api/helpers";

interface SelectedTemplateViewProps {
  agent: LibraryAgent;
  presetID: string;
  onDelete?: (presetID: string) => void;
  onCreateRun?: (runId: string) => void;
  onCreateSchedule?: (scheduleId: string) => void;
}

export function SelectedTemplateView({
  agent,
  presetID,
  onDelete,
  onCreateRun: _onCreateRun,
  onCreateSchedule: _onCreateSchedule,
}: SelectedTemplateViewProps) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [isDeleting, setIsDeleting] = useState(false);

  const templateOrTrigger = agent.trigger_setup_info ? "Trigger" : "Template";

  const presetQuery = useGetV2GetASpecificPreset(presetID, {
    query: {
      enabled: !!agent.graph_id && !!presetID,
      // select: okData,
    },
  });
  const preset = useMemo(() => okData(presetQuery.data), [presetQuery.data]);

  // Delete preset mutation
  const deleteTemplateMutation = useDeleteV2DeleteAPreset({
    mutation: {
      onSuccess: () => {
        toast({
          title: `${templateOrTrigger} deleted successfully`,
          variant: "default",
        });
        // Invalidate presets list
        queryClient.invalidateQueries({
          queryKey: getGetV2ListPresetsQueryKey({ graph_id: agent.graph_id }),
        });
        setIsDeleting(false);
      },
      onError: (error) => {
        toast({
          title: `Failed to delete ${templateOrTrigger.toLowerCase()}`,
          description: String(error),
          variant: "destructive",
        });
        setIsDeleting(false);
      },
    },
  });

  const doDeleteTemplate = async () => {
    setIsDeleting(true);
    deleteTemplateMutation.mutate({ presetId: presetID });
  };

  // Toggle trigger active status mutation
  const toggleTriggerStatusMutation = usePatchV2UpdateAnExistingPreset({
    mutation: {
      onSuccess: (response) => {
        if (response.status === 200) {
          toast({
            title: `Trigger ${preset?.is_active ? "disabled" : "enabled"} successfully`,
            variant: "default",
          });
          // Invalidate preset queries to refresh data
          queryClient.invalidateQueries({
            queryKey: getGetV2ListPresetsQueryKey({ graph_id: agent.graph_id }),
          });
          queryClient.invalidateQueries({
            queryKey: getGetV2GetASpecificPresetQueryKey(presetID),
          });
        }
      },
      onError: (error) => {
        toast({
          title: `Failed to ${preset?.is_active ? "disable" : "enable"} trigger`,
          description: String(error),
          variant: "destructive",
        });
      },
    },
  });

  const doToggleTriggerStatus = () => {
    if (!preset) return;
    toggleTriggerStatusMutation.mutate({
      presetId: presetID,
      data: {
        is_active: !preset.is_active,
      },
    });
  };

  const onSave = useCallback(() => {
    // Invalidate preset queries to refresh data
    queryClient.invalidateQueries({
      queryKey: getGetV2ListPresetsQueryKey({ graph_id: agent.graph_id }),
    });
    queryClient.invalidateQueries({
      queryKey: getGetV2GetASpecificPresetQueryKey(presetID),
    });
  }, [queryClient, agent.graph_id, presetID]);

  const onCreateRun = useCallback(
    (execution: GraphExecutionMeta) => {
      // Invalidate runs list
      queryClient.invalidateQueries({
        queryKey: getGetV1ListGraphExecutionsQueryKey(agent.graph_id),
      });
      _onCreateRun?.(execution.id);
    },
    [queryClient, agent.graph_id, _onCreateRun],
  );

  const onCreateSchedule = useCallback(
    (schedule: GraphExecutionJobInfo) => {
      // Invalidate schedules list
      queryClient.invalidateQueries({
        queryKey: getGetV1ListExecutionSchedulesForAGraphQueryKey(
          agent.graph_id,
        ),
      });
      _onCreateSchedule?.(schedule.id);
    },
    [queryClient, agent.graph_id, _onCreateSchedule],
  );

  const isLoading = presetQuery.isLoading;
  const error = presetQuery.error;

  if (error) {
    return (
      <ErrorCard
        responseError={
          error
            ? {
                message: String(
                  (error as unknown as { message?: string })?.message ||
                    `Failed to load ${templateOrTrigger.toLowerCase()}`,
                ),
              }
            : undefined
        }
        httpError={
          (error as any)?.status
            ? {
                status: (error as any).status,
                statusText: (error as any).statusText,
              }
            : undefined
        }
        context="template"
      />
    );
  }

  if (isLoading && !preset) {
    return (
      <div className="flex-1 space-y-4">
        <Skeleton className="h-8 w-full" />
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-64 w-full" />
        <Skeleton className="h-32 w-full" />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <div>
        <div className="flex w-full items-center justify-between">
          <div className="flex w-full flex-col gap-0">
            <div className="flex flex-col gap-2">
              <Text variant="h2" className="!text-2xl font-bold">
                {preset?.name || "Loading..."}
              </Text>
              {/* <Text variant="body-medium" className="!text-zinc-500">
                {templateOrTrigger} • {agent.name}
              </Text> */}
            </div>
          </div>
          {preset ? (
            <div className="flex gap-2">
              {!agent.has_external_trigger ? (
                <RunAgentModal
                  triggerSlot={
                    <Button
                      variant="primary"
                      size="small"
                      leftIcon={<PlayIcon size={16} />}
                    >
                      Run {templateOrTrigger}
                    </Button>
                  }
                  agent={agent}
                  initialInputValues={preset.inputs || {}}
                  initialInputCredentials={preset.credentials || {}}
                  initialPresetName={preset.name}
                  initialPresetDescription={preset.description}
                  onRunCreated={onCreateRun}
                  onScheduleCreated={onCreateSchedule}
                />
              ) : null}
              <RunAgentModal
                triggerSlot={
                  <Button
                    variant="secondary"
                    size="small"
                    leftIcon={<PencilIcon size={16} />}
                  >
                    Edit
                  </Button>
                }
                agent={agent}
                editMode={{
                  preset,
                  onSaved: onSave,
                }}
              />
              {/* Enable/Disable Trigger Button - only for triggered presets */}
              {preset.webhook && (
                <Button
                  variant={preset.is_active ? "destructive" : "primary"}
                  size="small"
                  onClick={doToggleTriggerStatus}
                  disabled={toggleTriggerStatusMutation.isPending}
                  leftIcon={
                    preset.is_active ? (
                      <StopIcon size={16} />
                    ) : (
                      <PlayIcon size={16} />
                    )
                  }
                >
                  {toggleTriggerStatusMutation.isPending
                    ? preset.is_active
                      ? "Disabling..."
                      : "Enabling..."
                    : preset.is_active
                      ? "Disable Trigger"
                      : "Enable Trigger"}
                </Button>
              )}
              <Button
                // TODO: add confirmation modal before deleting
                variant="destructive"
                size="small"
                onClick={() => {
                  doDeleteTemplate();
                  onDelete?.(presetID);
                }}
                disabled={isDeleting}
                leftIcon={<TrashIcon size={16} />}
              >
                {isDeleting ? "Deleting..." : "Delete"}
              </Button>
            </div>
          ) : null}
        </div>
      </div>

      <TabsLine defaultValue="input">
        <TabsLineList>
          <TabsLineTrigger value="input">Your input</TabsLineTrigger>
          <TabsLineTrigger value="details">
            {templateOrTrigger} details
          </TabsLineTrigger>
        </TabsLineList>

        <TabsLineContent value="input">
          <RunDetailCard>
            <div className="relative">
              <AgentInputsReadOnly
                agent={agent}
                inputs={preset?.inputs}
                credentialInputs={preset?.credentials}
              />
            </div>
          </RunDetailCard>
        </TabsLineContent>

        <TabsLineContent value="details">
          <RunDetailCard>
            {isLoading || !preset ? (
              <div className="text-neutral-500">Loading…</div>
            ) : (
              <div className="relative flex flex-col gap-8">
                <div className="flex flex-col gap-1.5">
                  <Text variant="body-medium" className="!text-black">
                    Name
                  </Text>
                  <p className="text-sm text-zinc-600">{preset.name}</p>
                </div>
                <div className="flex flex-col gap-1.5">
                  <Text variant="body-medium" className="!text-black">
                    Description
                  </Text>
                  <p className="text-sm text-zinc-600">
                    {preset.description || "No description provided"}
                  </p>
                </div>
                <div className="flex flex-col gap-1.5">
                  <Text variant="body-medium" className="!text-black">
                    Created
                  </Text>
                  <p className="text-sm text-zinc-600">
                    {new Date(preset.created_at).toLocaleDateString("en-US", {
                      year: "numeric",
                      month: "long",
                      day: "numeric",
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </p>
                </div>
                <div className="flex flex-col gap-1.5">
                  <Text variant="body-medium" className="!text-black">
                    Last Updated
                  </Text>
                  <p className="text-sm text-zinc-600">
                    {new Date(preset.updated_at).toLocaleDateString("en-US", {
                      year: "numeric",
                      month: "long",
                      day: "numeric",
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </p>
                </div>
              </div>
            )}
          </RunDetailCard>
        </TabsLineContent>
      </TabsLine>
    </div>
  );
}
