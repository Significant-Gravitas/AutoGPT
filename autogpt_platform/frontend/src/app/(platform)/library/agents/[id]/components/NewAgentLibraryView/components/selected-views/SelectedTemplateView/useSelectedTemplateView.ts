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
import { okData } from "@/app/api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import type { CredentialsMetaInput } from "@/lib/autogpt-server-api/types";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import {
  mergePresetInputs,
  splitPresetInputs,
} from "../SelectedTriggerView/helpers";

type Args = {
  templateId: string;
  graphId: string;
  onRunCreated?: (execution: GraphExecutionMeta) => void;
};

export function useSelectedTemplateView({
  templateId,
  graphId,
  onRunCreated,
}: Args) {
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const query = useGetV2GetASpecificPreset(templateId, {
    query: {
      enabled: !!templateId,
      select: okData,
    },
  });

  const [name, setName] = useState<string>("");
  const [description, setDescription] = useState<string>("");
  const [inputs, setInputs] = useState<Record<string, any>>({});
  // A detached triggered preset is filed here, so its inputs can carry the
  // trigger node's mask; keep the key so saving re-nests under the same one.
  const [triggerConfig, setTriggerConfig] = useState<Record<string, any>>({});
  const [maskKey, setMaskKey] = useState<string | null>(null);
  const [credentials, setCredentials] = useState<
    Record<string, CredentialsMetaInput>
  >({});

  useEffect(() => {
    if (query.data) {
      setName(query.data.name || "");
      setDescription(query.data.description || "");
      applyPresetInputs(query.data.inputs);
      setCredentials(query.data.credentials || {});
    }
  }, [query.data]);

  const updateMutation = usePatchV2UpdateAnExistingPreset({
    mutation: {
      onSuccess: (response) => {
        if (response.status === 200) {
          toast({
            title: "Template updated",
          });
          queryClient.invalidateQueries({
            queryKey: getGetV2GetASpecificPresetQueryKey(templateId),
          });
          queryClient.invalidateQueries({
            queryKey: getGetV2ListPresetsQueryKey({ graph_id: graphId }),
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
    mutation: {
      onSuccess: (response) => {
        if (response.status === 200) {
          const execution = okData(response);
          if (execution) {
            toast({
              title: "Task started",
            });
            queryClient.invalidateQueries({
              queryKey: getGetV1ListGraphExecutionsQueryKey(graphId),
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

    const storedInputs = mergePresetInputs({ inputs, triggerConfig, maskKey });

    const updateData: LibraryAgentPresetUpdatable = {};
    if (name !== (query.data.name || "")) {
      updateData.name = name;
    }

    if (description !== (query.data.description || "")) {
      updateData.description = description;
    }

    const inputsChanged =
      JSON.stringify(storedInputs) !== JSON.stringify(query.data.inputs || {});

    const credentialsChanged =
      JSON.stringify(credentials) !==
      JSON.stringify(query.data.credentials || {});

    if (inputsChanged || credentialsChanged) {
      updateData.inputs = storedInputs;
      updateData.credentials = credentials;
    }

    updateMutation.mutate({
      presetId: templateId,
      data: updateData,
    });
  }

  function handleStartTask() {
    if (!query.data) return;

    const storedInputs = mergePresetInputs({ inputs, triggerConfig, maskKey });

    const inputsChanged =
      JSON.stringify(storedInputs) !== JSON.stringify(query.data.inputs || {});

    const credentialsChanged =
      JSON.stringify(credentials) !==
      JSON.stringify(query.data.credentials || {});

    // Use changed unpersisted inputs if applicable
    executeMutation.mutate({
      presetId: templateId,
      data: {
        inputs: inputsChanged ? storedInputs : undefined,
        credential_inputs: credentialsChanged ? credentials : undefined,
      },
    });
  }

  function setInputValue(key: string, value: any) {
    setInputs((prev) => ({ ...prev, [key]: value }));
  }

  function setTriggerConfigValue(key: string, value: any) {
    setTriggerConfig((prev) => ({ ...prev, [key]: value }));
  }

  function applyPresetInputs(presetInputs: Record<string, any> | undefined) {
    const split = splitPresetInputs(presetInputs);
    setInputs(split.inputs);
    setTriggerConfig(split.triggerConfig);
    setMaskKey(split.maskKey);
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
      applyPresetInputs(query.data.inputs);
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
    triggerConfig,
    setTriggerConfigValue,
    hasTriggerConfig: maskKey !== null,
    credentials,
    setCredentialValue,
    handleSaveChanges,
    handleStartTask,
    isSaving: updateMutation.isPending,
    isStarting: executeMutation.isPending,
  } as const;
}
