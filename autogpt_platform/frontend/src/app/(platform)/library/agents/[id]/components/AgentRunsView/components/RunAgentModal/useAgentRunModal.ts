import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useState, useCallback, useMemo } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { isEmpty } from "@/lib/utils";
import {
  usePostV1ExecuteGraphAgent,
  getGetV1ListGraphExecutionsInfiniteQueryOptions,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import { usePostV2SetupTrigger } from "@/app/api/__generated__/endpoints/presets/presets";
import { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";

export type RunVariant =
  | "manual"
  | "schedule"
  | "automatic-trigger"
  | "manual-trigger";

interface UseAgentRunModalCallbacks {
  onRun?: (execution: GraphExecutionMeta) => void;
  onSetupTrigger?: (preset: LibraryAgentPreset) => void;
}

export function useAgentRunModal(
  agent: LibraryAgent,
  callbacks?: UseAgentRunModalCallbacks,
) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [isOpen, setIsOpen] = useState(false);
  const [inputValues, setInputValues] = useState<Record<string, any>>({});
  const [inputCredentials, setInputCredentials] = useState<Record<string, any>>(
    {},
  );
  const [presetName, setPresetName] = useState<string>("");
  const [presetDescription, setPresetDescription] = useState<string>("");
  // Determine the default run type based on agent capabilities
  const defaultRunType: RunVariant = agent.has_external_trigger
    ? "automatic-trigger"
    : "manual";

  // API mutations
  const executeGraphMutation = usePostV1ExecuteGraphAgent({
    mutation: {
      onSuccess: (response) => {
        if (response.status === 200) {
          toast({
            title: "Agent execution started",
          });
          callbacks?.onRun?.(response.data as unknown as GraphExecutionMeta);
          // Invalidate runs list for this graph
          queryClient.invalidateQueries({
            queryKey: getGetV1ListGraphExecutionsInfiniteQueryOptions(
              agent.graph_id,
            ).queryKey,
          });
          setIsOpen(false);
        }
      },
      onError: (error: any) => {
        toast({
          title: "❌ Failed to execute agent",
          description: error.message || "An unexpected error occurred.",
          variant: "destructive",
        });
      },
    },
  });

  const setupTriggerMutation = usePostV2SetupTrigger({
    mutation: {
      onSuccess: (response: any) => {
        if (response.status === 200) {
          toast({
            title: "Trigger setup complete",
          });
          callbacks?.onSetupTrigger?.(response.data);
          setIsOpen(false);
        }
      },
      onError: (error: any) => {
        toast({
          title: "❌ Failed to setup trigger",
          description: error.message || "An unexpected error occurred.",
          variant: "destructive",
        });
      },
    },
  });

  // Input schema validation
  const agentInputSchema = useMemo(
    () => agent.input_schema || { properties: {}, required: [] },
    [agent.input_schema],
  );

  const agentInputFields = useMemo(() => {
    if (
      !agentInputSchema ||
      typeof agentInputSchema !== "object" ||
      !("properties" in agentInputSchema) ||
      !agentInputSchema.properties
    ) {
      return {};
    }
    const properties = agentInputSchema.properties as Record<string, any>;
    return Object.fromEntries(
      Object.entries(properties).filter(
        ([_, subSchema]: [string, any]) => !subSchema.hidden,
      ),
    );
  }, [agentInputSchema]);

  const agentCredentialsInputFields = useMemo(() => {
    if (
      !agent.credentials_input_schema ||
      typeof agent.credentials_input_schema !== "object" ||
      !("properties" in agent.credentials_input_schema) ||
      !agent.credentials_input_schema.properties
    ) {
      return {} as Record<string, any>;
    }
    return agent.credentials_input_schema.properties as Record<string, any>;
  }, [agent.credentials_input_schema]);

  // Validation logic
  const [allRequiredInputsAreSetRaw, missingInputs] = useMemo(() => {
    const nonEmptyInputs = new Set(
      Object.keys(inputValues).filter((k) => !isEmpty(inputValues[k])),
    );
    const requiredInputs = new Set(
      (agentInputSchema.required as string[]) || [],
    );
    const missing = [...requiredInputs].filter(
      (input) => !nonEmptyInputs.has(input),
    );
    return [missing.length === 0, missing];
  }, [agentInputSchema.required, inputValues]);

  const [allCredentialsAreSet, missingCredentials] = useMemo(() => {
    const availableCredentials = new Set(Object.keys(inputCredentials));
    const allCredentials = new Set(
      Object.keys(agentCredentialsInputFields || {}) ?? [],
    );
    const missing = [...allCredentials].filter(
      (key) => !availableCredentials.has(key),
    );
    return [missing.length === 0, missing];
  }, [agentCredentialsInputFields, inputCredentials]);

  const credentialsRequired = useMemo(
    () => Object.keys(agentCredentialsInputFields || {}).length > 0,
    [agentCredentialsInputFields],
  );

  // Final readiness flag combining inputs + credentials when credentials are shown
  const allRequiredInputsAreSet = useMemo(
    () =>
      allRequiredInputsAreSetRaw &&
      (!credentialsRequired || allCredentialsAreSet),
    [allRequiredInputsAreSetRaw, credentialsRequired, allCredentialsAreSet],
  );

  const notifyMissingRequirements = useCallback(() => {
    const allMissingFields = missingInputs.concat(
      credentialsRequired && !allCredentialsAreSet
        ? missingCredentials.map((k) => `credentials:${k}`)
        : [],
    );

    toast({
      title: "⚠️ Missing required inputs",
      description: `Please provide: ${allMissingFields.map((k) => `"${k}"`).join(", ")}`,
      variant: "destructive",
    });
  }, [
    missingInputs,
    toast,
    credentialsRequired,
    allCredentialsAreSet,
    missingCredentials,
  ]);

  // Action handlers
  const handleRun = useCallback(() => {
    if (!allRequiredInputsAreSet) {
      notifyMissingRequirements();
      return;
    }

    if (defaultRunType === "automatic-trigger") {
      // Setup trigger
      if (!presetName.trim()) {
        toast({
          title: "⚠️ Trigger name required",
          description: "Please provide a name for your trigger.",
          variant: "destructive",
        });
        return;
      }

      setupTriggerMutation.mutate({
        data: {
          name: presetName,
          description: presetDescription || `Trigger for ${agent.name}`,
          graph_id: agent.graph_id,
          graph_version: agent.graph_version,
          trigger_config: inputValues,
          agent_credentials: inputCredentials,
        },
      });
    } else {
      // Manual execution
      executeGraphMutation.mutate({
        graphId: agent.graph_id,
        graphVersion: agent.graph_version,
        data: {
          inputs: inputValues,
          credentials_inputs: inputCredentials,
        },
      });
    }
  }, [
    allRequiredInputsAreSet,
    defaultRunType,
    inputValues,
    inputCredentials,
    agent,
    presetName,
    presetDescription,
    notifyMissingRequirements,
    setupTriggerMutation,
    executeGraphMutation,
    toast,
  ]);

  const hasInputFields = useMemo(() => {
    return Object.keys(agentInputFields).length > 0;
  }, [agentInputFields]);

  return {
    // UI state
    isOpen,
    setIsOpen,

    // Run mode
    defaultRunType,

    // Form: regular inputs
    inputValues,
    setInputValues,

    // Form: credentials
    inputCredentials,
    setInputCredentials,

    // Preset/trigger labels
    presetName,
    presetDescription,
    setPresetName,
    setPresetDescription,

    // Validation/readiness
    allRequiredInputsAreSet,
    missingInputs,

    // Schemas for rendering
    agentInputFields,
    agentCredentialsInputFields,
    hasInputFields,

    // Async states
    isExecuting: executeGraphMutation.isPending,
    isSettingUpTrigger: setupTriggerMutation.isPending,

    // Actions
    handleRun,
  };
}
