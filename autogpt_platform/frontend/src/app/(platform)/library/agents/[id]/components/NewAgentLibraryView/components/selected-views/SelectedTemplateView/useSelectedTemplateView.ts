"use client";

import { getGetV1ListGraphExecutionsQueryKey } from "@/app/api/__generated__/endpoints/graphs/graphs";
import {
  getGetV2GetASpecificPresetQueryKey,
  getGetV2ListPresetsQueryKey,
  useGetV2GetASpecificPreset,
  usePatchV2UpdateAnExistingPreset,
  usePostV2ExecuteAPreset,
} from "@/app/api/__generated__/endpoints/presets/presets";
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import type { LibraryAgentPresetUpdatable } from "@/app/api/__generated__/models/libraryAgentPresetUpdatable";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { okData } from "@/app/api/helpers";
import {
  getTenantRequestInit,
  getTeamScopedQueryKey,
} from "@/components/contextual/TeamPicker/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import type { CredentialsMetaInput } from "@/lib/autogpt-server-api/types";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

type Args = {
  templateId: string;
  agent: LibraryAgent;
  onRunCreated?: (execution: GraphExecutionMeta) => void;
};

export function useSelectedTemplateView({
  templateId,
  agent,
  onRunCreated,
}: Args) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const organizationId = agent.organization_id ?? null;
  const teamId = agent.team_id ?? null;
  const graphId = agent.graph_id;

  const query = useGetV2GetASpecificPreset(templateId, {
    request: getTenantRequestInit(organizationId, teamId),
    query: {
      queryKey: getTeamScopedQueryKey(
        getGetV2GetASpecificPresetQueryKey(templateId),
        organizationId,
        teamId,
      ),
      enabled: !!templateId,
      select: okData,
    },
  });

  const [name, setName] = useState<string>("");
  const [description, setDescription] = useState<string>("");
  const [inputs, setInputs] = useState<Record<string, any>>({});
  const [credentials, setCredentials] = useState<
    Record<string, CredentialsMetaInput>
  >({});

  useEffect(() => {
    if (query.data) {
      setName(query.data.name || "");
      setDescription(query.data.description || "");
      setInputs(query.data.inputs || {});
      setCredentials(query.data.credentials || {});
    }
  }, [query.data]);

  const updateMutation = usePatchV2UpdateAnExistingPreset({
    request: getTenantRequestInit(organizationId, teamId),
    mutation: {
      onSuccess: (response) => {
        if (response.status === 200) {
          toast({
            title: "Template updated",
          });
          queryClient.invalidateQueries({
            queryKey: getTeamScopedQueryKey(
              getGetV2GetASpecificPresetQueryKey(templateId),
              organizationId,
              teamId,
            ),
          });
          queryClient.invalidateQueries({
            queryKey: getTeamScopedQueryKey(
              getGetV2ListPresetsQueryKey({ graph_id: graphId }),
              organizationId,
              teamId,
            ),
          });
        }
      },
      onError: (error: any) => {
        toast({
          title: "Failed to update template",
          description: error.message || "An unexpected error occurred.",
          variant: "destructive",
        });
      },
    },
  });

  const executeMutation = usePostV2ExecuteAPreset({
    request: getTenantRequestInit(organizationId, teamId),
    mutation: {
      onSuccess: (response) => {
        if (response.status === 200) {
          const execution = okData(response);
          if (execution) {
            toast({
              title: "Task started",
            });
            queryClient.invalidateQueries({
              queryKey: getTeamScopedQueryKey(
                getGetV1ListGraphExecutionsQueryKey(graphId),
                organizationId,
                teamId,
              ),
            });
            onRunCreated?.(execution);
          }
        }
      },
      onError: (error: any) => {
        toast({
          title: "Failed to start task",
          description: error.message || "An unexpected error occurred.",
          variant: "destructive",
        });
      },
    },
  });

  function handleSaveChanges() {
    if (!query.data) return;

    const updateData: LibraryAgentPresetUpdatable = {};
    if (name !== (query.data.name || "")) {
      updateData.name = name;
    }

    if (description !== (query.data.description || "")) {
      updateData.description = description;
    }

    const inputsChanged =
      JSON.stringify(inputs) !== JSON.stringify(query.data.inputs || {});

    const credentialsChanged =
      JSON.stringify(credentials) !==
      JSON.stringify(query.data.credentials || {});

    if (inputsChanged || credentialsChanged) {
      updateData.inputs = inputs;
      updateData.credentials = credentials;
    }

    updateMutation.mutate({
      presetId: templateId,
      data: updateData,
    });
  }

  function handleStartTask() {
    if (!query.data) return;

    const inputsChanged =
      JSON.stringify(inputs) !== JSON.stringify(query.data.inputs || {});

    const credentialsChanged =
      JSON.stringify(credentials) !==
      JSON.stringify(query.data.credentials || {});

    // Use changed unpersisted inputs if applicable
    executeMutation.mutate({
      presetId: templateId,
      data: {
        inputs: inputsChanged ? inputs : undefined,
        credential_inputs: credentialsChanged ? credentials : undefined,
      },
    });
  }

  function setInputValue(key: string, value: any) {
    setInputs((prev) => ({ ...prev, [key]: value }));
  }

  function setCredentialValue(key: string, value: CredentialsMetaInput) {
    setCredentials((prev) => ({ ...prev, [key]: value }));
  }

  const httpError =
    query.isSuccess && !query.data
      ? { status: 404, statusText: "Not found" }
      : undefined;

  useEffect(() => {
    if (updateMutation.isSuccess && query.data) {
      setName(query.data.name || "");
      setDescription(query.data.description || "");
      setInputs(query.data.inputs || {});
      setCredentials(query.data.credentials || {});
    }
  }, [updateMutation.isSuccess, query.data]);

  return {
    template: query.data,
    isLoading: query.isLoading,
    error: query.error || httpError,
    name,
    setName,
    description,
    setDescription,
    inputs,
    setInputValue,
    credentials,
    setCredentialValue,
    handleSaveChanges,
    handleStartTask,
    isSaving: updateMutation.isPending,
    isStarting: executeMutation.isPending,
  } as const;
}
