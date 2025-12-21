"use client";

import {
  getGetV2GetASpecificPresetQueryKey,
  getGetV2ListPresetsQueryKey,
  useGetV2GetASpecificPreset,
  usePatchV2UpdateAnExistingPreset,
} from "@/app/api/__generated__/endpoints/presets/presets";
import type { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";
import type { LibraryAgentPresetUpdatable } from "@/app/api/__generated__/models/libraryAgentPresetUpdatable";
import { okData } from "@/app/api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import type { CredentialsMetaInput } from "@/lib/autogpt-server-api/types";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

type Args = {
  triggerId: string;
  graphId: string;
};

export function useSelectedTriggerView({ triggerId, graphId }: Args) {
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const query = useGetV2GetASpecificPreset(triggerId, {
    query: {
      enabled: !!triggerId,
      select: (res) => okData<LibraryAgentPreset>(res),
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
    mutation: {
      onSuccess: (response) => {
        if (response.status === 200) {
          toast({
            title: "Trigger updated",
          });
          queryClient.invalidateQueries({
            queryKey: getGetV2GetASpecificPresetQueryKey(triggerId),
          });
          queryClient.invalidateQueries({
            queryKey: getGetV2ListPresetsQueryKey({ graph_id: graphId }),
          });
        }
      },
      onError: (error: any) => {
        toast({
          title: "Failed to update trigger",
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
      presetId: triggerId,
      data: updateData,
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
    trigger: query.data,
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
    isSaving: updateMutation.isPending,
  } as const;
}
