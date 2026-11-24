"use client";

import {
  getGetV2GetASpecificPresetQueryKey,
  getGetV2ListPresetsQueryKey,
  useGetV2GetASpecificPreset,
  usePatchV2UpdateAnExistingPreset,
} from "@/app/api/__generated__/endpoints/presets/presets";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import type { LibraryAgentPresetUpdatable } from "@/app/api/__generated__/models/libraryAgentPresetUpdatable";
import { okData } from "@/app/api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import type { CredentialsMetaInput } from "@/lib/autogpt-server-api/types";
import { retryUnlessClientError } from "../../../helpers";
import { withCalendarDates } from "../../modals/RunAgentInputs/helpers";
import { mergeTriggerPresetInputs, splitTriggerPresetInputs } from "./helpers";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

type Args = {
  triggerId: string;
  graphId: string;
  // The schemas the form shows, so a stored date is read as the day it shows.
  inputSchema?: LibraryAgent["input_schema"];
  triggerConfigSchema?: NonNullable<
    LibraryAgent["trigger_setup_info"]
  >["config_schema"];
};

export function useSelectedTriggerView({
  triggerId,
  graphId,
  inputSchema,
  triggerConfigSchema,
}: Args) {
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const query = useGetV2GetASpecificPreset(triggerId, {
    query: {
      enabled: !!triggerId,
      select: okData,
      retry: retryUnlessClientError,
    },
  });

  const [name, setName] = useState<string>("");
  const [description, setDescription] = useState<string>("");
  const [inputs, setInputs] = useState<Record<string, any>>({});
  const [triggerConfig, setTriggerConfig] = useState<Record<string, any>>({});
  const [maskKey, setMaskKey] = useState<string | null>(null);
  const [credentials, setCredentials] = useState<
    Record<string, CredentialsMetaInput>
  >({});

  // The stored inputs, split into the graph's and the trigger's, each date as
  // the day the form shows.
  function applyPresetInputs(args: {
    presetInputs: Record<string, unknown> | undefined;
    inputSchema: Args["inputSchema"];
    triggerConfigSchema: Args["triggerConfigSchema"];
  }) {
    const split = splitTriggerPresetInputs(args.presetInputs);
    setInputs(withCalendarDates(args.inputSchema, split.inputs));
    setTriggerConfig(
      withCalendarDates(args.triggerConfigSchema, split.triggerConfig),
    );
    setMaskKey(split.maskKey);
  }

  useEffect(() => {
    if (query.data) {
      setName(query.data.name || "");
      setDescription(query.data.description || "");
      applyPresetInputs({
        presetInputs: query.data.inputs,
        inputSchema,
        triggerConfigSchema,
      });
      setCredentials(query.data.credentials || {});
    }
  }, [query.data, inputSchema, triggerConfigSchema]);

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

    // Without a mask key the backend reads every stored input as trigger config,
    // so a graph input saved now would land inside it.
    if (!maskKey && Object.keys(inputs).length > 0) {
      toast({
        title: "Failed to update trigger",
        description:
          "This trigger was saved in an older format. " +
          "Save its trigger settings first, then set its inputs.",
        variant: "destructive",
      });
      return;
    }

    const updateData: LibraryAgentPresetUpdatable = {};
    if (name !== (query.data.name || "")) {
      updateData.name = name;
    }

    if (description !== (query.data.description || "")) {
      updateData.description = description;
    }

    // Send back the stored shape, so update_triggered_preset finds the trigger
    // config where it keeps it.
    const mergedInputs = mergeTriggerPresetInputs({
      inputs,
      triggerConfig,
      maskKey,
    });

    const inputsChanged =
      JSON.stringify(mergedInputs) !== JSON.stringify(query.data.inputs || {});

    const credentialsChanged =
      JSON.stringify(credentials) !==
      JSON.stringify(query.data.credentials || {});

    if (inputsChanged || credentialsChanged) {
      updateData.inputs = mergedInputs;
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

  function setTriggerConfigValue(key: string, value: any) {
    setTriggerConfig((prev) => ({ ...prev, [key]: value }));
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
      applyPresetInputs({
        presetInputs: query.data.inputs,
        inputSchema,
        triggerConfigSchema,
      });
      setCredentials(query.data.credentials || {});
    }
  }, [updateMutation.isSuccess, query.data, inputSchema, triggerConfigSchema]);

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
    triggerConfig,
    setTriggerConfigValue,
    credentials,
    setCredentialValue,
    handleSaveChanges,
    isSaving: updateMutation.isPending,
  } as const;
}
