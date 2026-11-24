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
import {
  CredentialFieldSchema,
  hasOwnCredentialFor,
} from "@/components/contextual/CredentialsInput/components/CredentialsGroupedView/node-credentials";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { CredentialsProvidersContext } from "@/providers/agent-credentials/credentials-provider";
import { useQueryClient } from "@tanstack/react-query";
import {
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { getVisibleFields } from "../AgentInputsReadOnly/helpers";
import { withCalendarDates } from "../RunAgentInputs/helpers";
import { showExecutionErrorToast } from "./errorHelpers";
import { getDefaultRunCredentials } from "./run-credentials";
import { useSavedNodeCredentials } from "./useSavedNodeCredentials";

export type RunVariant =
  | "manual"
  | "schedule"
  | "automatic-trigger"
  | "manual-trigger";

interface UseAgentRunModalCallbacks {
  onRun?: (execution: GraphExecutionMeta) => void;
  onSetupTrigger?: (preset: LibraryAgentPreset) => void;
  initialInputValues?: Record<string, any>;
  initialTriggerConfigValues?: Record<string, any>;
  initialInputCredentials?: Record<string, any>;
}

export function useAgentRunModal(
  agent: LibraryAgent,
  callbacks?: UseAgentRunModalCallbacks,
) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [isOpen, setIsOpen] = useState(false);
  const [inputValues, setInputValues] = useState<Record<string, any>>(() =>
    withCalendarDates(agent.input_schema, callbacks?.initialInputValues),
  );
  const [inputCredentials, setInputCredentials] = useState<Record<string, any>>(
    callbacks?.initialInputCredentials || {},
  );
  // For triggered agents the form is split in two: `inputValues` holds the
  // graph's regular inputs (sent as `constant_inputs` when setting up the
  // trigger), and these hold the trigger node's config (sent as
  // `trigger_config`). Seeded from `initialTriggerConfigValues` when editing.
  const [triggerConfigValues, setTriggerConfigValues] = useState<
    Record<string, any>
  >(() =>
    withCalendarDates(
      agent.trigger_setup_info?.config_schema,
      callbacks?.initialTriggerConfigValues,
    ),
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
    setInputValues(
      withCalendarDates(agent.input_schema, callbacks?.initialInputValues),
    );
    setInputCredentials(callbacks?.initialInputCredentials || {});
    setTriggerConfigValues(
      withCalendarDates(
        agent.trigger_setup_info?.config_schema,
        callbacks?.initialTriggerConfigValues,
      ),
    );
  }, [
    agent.input_schema,
    agent.trigger_setup_info?.config_schema,
    callbacks?.initialInputValues,
    callbacks?.initialTriggerConfigValues,
    callbacks?.initialInputCredentials,
  ]);

  const allProviders = useContext(CredentialsProvidersContext);
  const credentialFieldSchemas = Object.values(
    (agent.credentials_input_schema?.properties ?? {}) as Record<
      string,
      CredentialFieldSchema
    >,
  );
  // A step can only hold one of the user's own keys, so the steps' keys only
  // matter to a user with a key of their own, and a template brings its own.
  const hasOwnCredential = hasOwnCredentialFor(
    credentialFieldSchemas,
    allProviders,
  );
  const needsNodeCredentials =
    hasOwnCredential && !callbacks?.initialInputCredentials;
  // Only the graph's owner gets the keys picked on its steps back. They are
  // read with the page for a user with a key of their own, so they are there
  // before the form opens, and otherwise once the form opens, where the user
  // may add their first key.
  const {
    nodeCredentials,
    isLoadingNodeCredentials,
    hasNodeCredentialsError,
    isRetryingNodeCredentials,
    retryNodeCredentials,
  } = useSavedNodeCredentials(
    agent,
    Boolean(agent.can_access_graph) &&
      credentialFieldSchemas.length > 0 &&
      (hasOwnCredential || isOpen),
  );
  const isResolvingCredentials =
    isLoadingNodeCredentials && needsNodeCredentials;
  const nodeCredentialsUnknown =
    hasNodeCredentialsError && needsNodeCredentials;

  // Start each credential on the key its steps use, else the platform's key
  useEffect(() => {
    if (!allProviders || !agent.credentials_input_schema?.properties) return;
    if (callbacks?.initialInputCredentials) {
      hasInitializedSystemCreds.current = true;
      return;
    }
    if (hasInitializedSystemCreds.current || isResolvingCredentials) return;

    const properties = agent.credentials_input_schema.properties as Record<
      string,
      CredentialFieldSchema
    >;

    setInputCredentials((currentCreds) => {
      const credsToAdd = getDefaultRunCredentials({
        properties,
        currentCredentials: currentCreds,
        allProviders,
        nodeCredentials,
        nodeCredentialsUnknown,
      });
      if (Object.keys(credsToAdd).length === 0) return currentCreds;

      // While the steps' keys are unknown the form isn't settled: once they
      // load, the fields left for them still start on the step's key.
      if (!nodeCredentialsUnknown) hasInitializedSystemCreds.current = true;
      return {
        ...currentCreds,
        ...credsToAdd,
      };
    });
  }, [
    allProviders,
    agent.credentials_input_schema,
    callbacks?.initialInputCredentials,
    isResolvingCredentials,
    nodeCredentials,
    nodeCredentialsUnknown,
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
            title: "Workflow execution started",
          });
          queryClient.invalidateQueries({
            queryKey: getGetV1ListGraphExecutionsQueryKey(agent.graph_id),
          });
          callbacks?.onRun?.(response.data);
          setIsOpen(false);
        }
      },
      onError: (error: any) => {
        showExecutionErrorToast(toast, error, {
          graph_id: agent.graph_id,
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

  // Primary input fields: the agent's regular graph inputs — consistent for
  // both triggered and non-triggered agents.
  const agentInputSchema = useMemo(
    () => agent.input_schema || { properties: {}, required: [] },
    [agent.input_schema],
  );

  const agentInputFields = useMemo(
    () => getVisibleFields(agentInputSchema),
    [agentInputSchema],
  );

  // Triggered agents additionally have the trigger node's config, collected in
  // its own conditional section and sent as `trigger_config`.
  const triggerConfigSchema = useMemo(
    () => agent.trigger_setup_info?.config_schema,
    [agent.trigger_setup_info],
  );

  const triggerConfigFields = useMemo(
    () => getVisibleFields(triggerConfigSchema),
    [triggerConfigSchema],
  );

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
    const missingFor = (
      schema: { required?: unknown } | undefined,
      values: Record<string, any>,
    ) => {
      const required = (schema?.required as string[]) || [];
      return required.filter((field) => !isFilled(values[field]));
    };

    const missing = [
      ...missingFor(agentInputSchema, inputValues),
      ...missingFor(triggerConfigSchema, triggerConfigValues),
    ];
    return [missing.length === 0, missing];
  }, [agentInputSchema, inputValues, triggerConfigSchema, triggerConfigValues]);

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
      !isResolvingCredentials &&
      (!credentialsRequired || allCredentialsAreSet),
    [
      allRequiredInputsAreSetRaw,
      isResolvingCredentials,
      credentialsRequired,
      allCredentialsAreSet,
    ],
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
          trigger_config: triggerConfigValues,
          constant_inputs: inputValues,
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
    triggerConfigValues,
    inputCredentials,
    agent,
    presetName,
    presetDescription,
    notifyMissingRequirements,
    setupTriggerMutation,
    executeGraphMutation,
    toast,
  ]);

  const handleSimulate = useCallback(() => {
    executeGraphMutation.mutate({
      graphId: agent.graph_id,
      graphVersion: agent.graph_version,
      data: {
        inputs: inputValues,
        credentials_inputs: {},
        source: "library",
        dry_run: true,
      },
    });
  }, [agent, inputValues, executeGraphMutation]);

  const hasInputFields = useMemo(() => {
    return (
      Object.keys(agentInputFields).length > 0 ||
      Object.keys(triggerConfigFields).length > 0
    );
  }, [agentInputFields, triggerConfigFields]);

  return {
    isOpen,
    setIsOpen,
    defaultRunType: defaultRunType as RunVariant,
    inputValues,
    setInputValues,
    triggerConfigValues,
    setTriggerConfigValues,
    inputCredentials,
    setInputCredentials,
    presetName,
    presetDescription,
    setPresetName,
    setPresetDescription,
    allRequiredInputsAreSet,
    missingInputs,
    agentInputFields,
    triggerConfigFields,
    agentCredentialsInputFields,
    nodeCredentials,
    isResolvingCredentials,
    nodeCredentialsUnknown,
    isRetryingNodeCredentials,
    retryNodeCredentials,
    hasInputFields,
    isExecuting: executeGraphMutation.isPending,
    isSettingUpTrigger: setupTriggerMutation.isPending,
    handleRun,
    handleSimulate,
  };
}

// Mirror the backend's `_is_filled`: only null/undefined, "", {}, and [] count
// as missing — falsy primitives like 0 and false are valid provided values.
export function isFilled(value: unknown): boolean {
  if (value === null || value === undefined || value === "") return false;
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === "object") return Object.keys(value).length > 0;
  return true;
}
