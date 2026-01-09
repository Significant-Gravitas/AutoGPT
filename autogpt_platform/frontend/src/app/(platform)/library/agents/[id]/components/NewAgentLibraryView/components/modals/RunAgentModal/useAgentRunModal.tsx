import {
  getGetV1ListGraphExecutionsQueryKey,
  usePostV1ExecuteGraphAgent,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import {
  getGetV2ListPresetsQueryKey,
  usePostV2SetupTrigger,
} from "@/app/api/__generated__/endpoints/presets/presets";
import { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { isEmpty } from "@/lib/utils";
import { analytics } from "@/services/analytics";
import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useState } from "react";
import { showExecutionErrorToast } from "./errorHelpers";

export type RunVariant =
  | "manual"
  | "schedule"
  | "automatic-trigger"
  | "manual-trigger";

interface UseAgentRunModalCallbacks {
  onRun?: (execution: GraphExecutionMeta) => void;
  onSetupTrigger?: (preset: LibraryAgentPreset) => void;
  initialInputValues?: Record<string, any>;
  initialInputCredentials?: Record<string, any>;
}

export function useAgentRunModal(
  agent: LibraryAgent,
  callbacks?: UseAgentRunModalCallbacks,
) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [isOpen, setIsOpen] = useState(false);
  const [inputValues, setInputValues] = useState<Record<string, any>>(
    callbacks?.initialInputValues || {},
  );
  const [inputCredentials, setInputCredentials] = useState<Record<string, any>>(
    callbacks?.initialInputCredentials || {},
  );
  const [presetName, setPresetName] = useState<string>("");
  const [presetDescription, setPresetDescription] = useState<string>("");

  // Determine the default run type based on agent capabilities
  const defaultRunType: RunVariant = agent.trigger_setup_info
    ? agent.trigger_setup_info.credentials_input_name
      ? "automatic-trigger"
      : "manual-trigger"
    : "manual";

  // Update input values/credentials if template is selected/unselected
  useEffect(() => {
    setInputValues(callbacks?.initialInputValues || {});
    setInputCredentials(callbacks?.initialInputCredentials || {});
  }, [callbacks?.initialInputValues, callbacks?.initialInputCredentials]);

  // API mutations
  const executeGraphMutation = usePostV1ExecuteGraphAgent({
    mutation: {
      onSuccess: (response) => {
        if (response.status === 200) {
          toast({
            title: "Agent execution started",
          });
          // Invalidate runs list for this graph
          queryClient.invalidateQueries({
            queryKey: getGetV1ListGraphExecutionsQueryKey(agent.graph_id),
          });
          callbacks?.onRun?.(response.data);
          analytics.sendDatafastEvent("run_agent", {
            name: agent.name,
            id: agent.graph_id,
          });
          setIsOpen(false);
        }
      },
      onError: (error: any) => {
        showExecutionErrorToast(toast, error, {
          graph_id: agent.graph_id,
          graph_version: agent.graph_version,
        });
      },
    },
  });

  const setupTriggerMutation = usePostV2SetupTrigger({
    mutation: {
      onSuccess: (response) => {
        if (response.status === 200) {
          toast({
            title: "Trigger setup complete",
          });
          queryClient.invalidateQueries({
            queryKey: getGetV2ListPresetsQueryKey({ graph_id: agent.graph_id }),
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

  // Input schema validation (use trigger schema for triggered agents)
  const agentInputSchema = useMemo(() => {
    if (agent.trigger_setup_info?.config_schema) {
      return agent.trigger_setup_info.config_schema;
    }
    return agent.input_schema || { properties: {}, required: [] };
  }, [agent.input_schema, agent.trigger_setup_info]);

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
    // Only check required credentials from schema, not all properties
    // Credentials marked as optional in node metadata won't be in the required array
    const requiredCredentials = new Set(
      (agent.credentials_input_schema?.required as string[]) || [],
    );

    // Check if required credentials have valid id (not just key existence)
    // A credential is valid only if it has an id field set
    const missing = [...requiredCredentials].filter((key) => {
      const cred = inputCredentials[key];
      return !cred || !cred.id;
    });

    return [missing.length === 0, missing];
  }, [agent.credentials_input_schema, inputCredentials]);

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

    if (
      defaultRunType === "automatic-trigger" ||
      defaultRunType === "manual-trigger"
    ) {
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
      // Filter out incomplete credentials (optional ones not selected)
      // Only send credentials that have a valid id field
      const validCredentials = Object.fromEntries(
        Object.entries(inputCredentials).filter(([_, cred]) => cred && cred.id),
      );

      executeGraphMutation.mutate({
        graphId: agent.graph_id,
        graphVersion: agent.graph_version,
        data: {
          inputs: inputValues,
          credentials_inputs: validCredentials,
          source: "library",
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
    defaultRunType: defaultRunType as RunVariant,

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
