"use client";

import React, { useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import {
  PlayIcon,
  PencilIcon,
  TrashIcon,
  CalendarIcon,
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
  usePostV2ExecuteAPreset,
  getGetV2ListPresetsQueryKey,
  useGetV2GetASpecificPreset,
  getGetV2GetASpecificPresetQueryKey,
} from "@/app/api/__generated__/endpoints/presets/presets";
import { getGetV1ListGraphExecutionsInfiniteQueryOptions } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { RunDetailCard } from "../RunDetailCard/RunDetailCard";
import { Skeleton } from "@/components/__legacy__/ui/skeleton";
import { AgentInputsReadOnly } from "../AgentInputsReadOnly/AgentInputsReadOnly";
import { EditTemplateModal } from "./components/EditTemplateModal";
import { okData } from "@/app/api/helpers";

interface Props {
  agent: LibraryAgent;
  presetID: string;
  onDelete?: (presetID: string) => void;
  onRun?: (presetID: string) => void;
  onCreateSchedule?: (presetID: string) => void;
}

export function SelectedTemplateView({
  agent,
  presetID,
  onDelete,
  onRun,
  onCreateSchedule,
}: Props) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [isDeleting, setIsDeleting] = useState(false);
  const [isRunning, setIsRunning] = useState(false);

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

  // Execute preset mutation
  const executeTemplateMutation = usePostV2ExecuteAPreset({
    mutation: {
      onSuccess: (response) => {
        if (response.status === 200) {
          toast({
            title: `${templateOrTrigger} execution started`,
            variant: "default",
          });
          // Invalidate runs list for this graph
          queryClient.invalidateQueries({
            queryKey: getGetV1ListGraphExecutionsInfiniteQueryOptions(
              agent.graph_id,
            ).queryKey,
          });
        }
        setIsRunning(false);
      },
      onError: (error) => {
        toast({
          title: `Failed to run ${templateOrTrigger.toLowerCase()}`,
          description: String(error),
          variant: "destructive",
        });
        setIsRunning(false);
      },
    },
  });

  const handleDeleteTemplate = async () => {
    setIsDeleting(true);
    deleteTemplateMutation.mutate({ presetId: presetID });
  };

  const handleRunTemplate = async () => {
    setIsRunning(true);
    executeTemplateMutation.mutate({
      presetId: presetID,
      data: {
        // inputs: {},
        // credential_inputs: {},
      },
    });
  };

  const handleTemplateSaved = () => {
    // Invalidate presets list to refresh data
    queryClient.invalidateQueries({
      queryKey: getGetV2ListPresetsQueryKey({ graph_id: agent.graph_id }),
    });
    queryClient.invalidateQueries({
      queryKey: getGetV2GetASpecificPresetQueryKey(presetID),
    });
  };

  const handleCreateSchedule = () => {
    // TODO: Implement schedule creation from preset
    toast({
      title: "Schedule creation",
      description:
        "Schedule creation from template will be implemented in the next phase",
      variant: "default",
    });
  };

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
                <Button
                  variant="primary"
                  size="small"
                  onClick={() => {
                    handleRunTemplate();
                    onRun?.(presetID);
                  }}
                  disabled={isRunning || isDeleting}
                  leftIcon={<PlayIcon size={16} />}
                >
                  {isRunning ? "Running..." : `Run ${templateOrTrigger}`}
                </Button>
              ) : null}
              <EditTemplateModal
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
                preset={preset}
                onSaved={handleTemplateSaved}
              />
              <Button
                variant="secondary"
                size="small"
                onClick={() => {
                  handleCreateSchedule();
                  onCreateSchedule?.(presetID);
                }}
                leftIcon={<CalendarIcon size={16} />}
              >
                Schedule
              </Button>
              <Button
                variant="destructive"
                size="small"
                onClick={() => {
                  handleDeleteTemplate();
                  onDelete?.(presetID);
                }}
                disabled={isRunning || isDeleting}
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
