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
import { CredentialsProvidersContext } from "@/providers/agent-credentials/credentials-provider";
import { analytics } from "@/services/analytics";
import { useQueryClient } from "@tanstack/react-query";
import {
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { getSystemCredentials } from "../../../../../../../../../../components/contextual/CredentialsInput/helpers";
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
  const hasInitializedSystemCreds = useRef(false);

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

  const allProviders = useContext(CredentialsProvidersContext);

  // Initialize credentials with default system credentials
  useEffect(() => {
    if (!allProviders || !agent.credentials_input_schema?.properties) return;
    if (callbacks?.initialInputCredentials) {
      hasInitializedSystemCreds.current = true;
      return;
    }
    if (hasInitializedSystemCreds.current) return;

    const properties = agent.credentials_input_schema.properties as Record<
      string,
      any
    >;

    setInputCredentials((currentCreds) => {
      const credsToAdd: Record<string, any> = {};

      for (const [key, schema] of Object.entries(properties)) {
        if (currentCreds[key]) continue;

        const providerNames = schema.credentials_provider || [];
        const supportedTypes = schema.credentials_types || [];
        const requiredScopes = schema.credentials_scopes;

        for (const providerName of providerNames) {
          const providerData = allProviders[providerName];
          if (!providerData) continue;

          const systemCreds = getSystemCredentials(
            providerData.savedCredentials ?? [],
          );
          const matchingSystemCreds = systemCreds.filter((cred) => {
            if (!supportedTypes.includes(cred.type)) return false;

            if (
              cred.type === "oauth2" &&
              requiredScopes &&
              requiredScopes.length > 0
            ) {
              const grantedScopes = new Set(cred.scopes || []);
              const hasAllRequiredScopes = requiredScopes.every(
                (scope: string) => grantedScopes.has(scope),
              );
              if (!hasAllRequiredScopes) return false;
            }

            return true;
          });

          if (matchingSystemCreds.length === 1) {
            const systemCred = matchingSystemCreds[0];
            credsToAdd[key] = {
              id: systemCred.id,
              type: systemCred.type,
              provider: providerName,
              title: systemCred.title,
            };
            break;
          }
        }
      }

      if (Object.keys(credsToAdd).length > 0) {
        hasInitializedSystemCreds.current = true;
        return {
          ...currentCreds,
          ...credsToAdd,
        };
      }

      return currentCreds;
    });
  }, [
    allProviders,
    agent.credentials_input_schema,
    callbacks?.initialInputCredentials,
  ]);

  // Reset initialization flag when modal closes/opens or agent changes
  useEffect(() => {
    hasInitializedSystemCreds.current = false;
  }, [isOpen, agent.graph_id]);

  // API mutations
  const executeGraphMutation = usePostV1ExecuteGraphAgent({
    mutation: {
      onSuccess: (response) => {
        if (response.status === 200) {
          toast({
            title: "Agent execution started",
          });
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
    const requiredCredentials = new Set(
      (agent.credentials_input_schema?.required as string[]) || [],
    );

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
    isOpen,
    setIsOpen,
    defaultRunType: defaultRunType as RunVariant,
    inputValues,
    setInputValues,
    inputCredentials,
    setInputCredentials,
    presetName,
    presetDescription,
    setPresetName,
    setPresetDescription,
    allRequiredInputsAreSet,
    missingInputs,
    agentInputFields,
    agentCredentialsInputFields,
    hasInputFields,
    isExecuting: executeGraphMutation.isPending,
    isSettingUpTrigger: setupTriggerMutation.isPending,
    handleRun,
  };
}
