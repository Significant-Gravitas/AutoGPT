import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useState, useCallback, useMemo } from "react";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { isEmpty } from "@/lib/utils";
import { usePostV1ExecuteGraphAgent } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { usePostV1CreateExecutionSchedule as useCreateSchedule } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { usePostV2SetupTrigger } from "@/app/api/__generated__/endpoints/presets/presets";
import { ExecuteGraphResponse } from "@/app/api/__generated__/models/executeGraphResponse";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";

export type RunVariant =
  | "manual"
  | "schedule"
  | "automatic-trigger"
  | "manual-trigger";

interface UseAgentRunModalCallbacks {
  onRun?: (execution: ExecuteGraphResponse) => void;
  onCreateSchedule?: (schedule: GraphExecutionJobInfo) => void;
  onSetupTrigger?: (preset: LibraryAgentPreset) => void;
}

export function useAgentRunModal(
  agent: LibraryAgent,
  callbacks?: UseAgentRunModalCallbacks,
) {
  const { toast } = useToast();
  const [isOpen, setIsOpen] = useState(false);
  const [showScheduleView, setShowScheduleView] = useState(false);
  const [inputValues, setInputValues] = useState<Record<string, any>>({});
  const [inputCredentials, setInputCredentials] = useState<Record<string, any>>(
    {},
  );
  const [presetName, setPresetName] = useState<string>("");
  const [presetDescription, setPresetDescription] = useState<string>("");
  const defaultScheduleName = useMemo(() => `Run ${agent.name}`, [agent.name]);
  const [scheduleName, setScheduleName] = useState(defaultScheduleName);
  const [cronExpression, setCronExpression] = useState("0 9 * * 1");

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
          callbacks?.onRun?.(response.data);
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

  const createScheduleMutation = useCreateSchedule({
    mutation: {
      onSuccess: (response) => {
        if (response.status === 200) {
          toast({
            title: "Schedule created",
          });
          callbacks?.onCreateSchedule?.(response.data);
          setIsOpen(false);
        }
      },
      onError: (error: any) => {
        toast({
          title: "❌ Failed to create schedule",
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

  const notifyMissingRequirements = useCallback(
    (needScheduleName: boolean = false) => {
      const allMissingFields = (
        needScheduleName && !scheduleName ? ["schedule_name"] : []
      )
        .concat(missingInputs)
        .concat(
          credentialsRequired && !allCredentialsAreSet
            ? missingCredentials.map((k) => `credentials:${k}`)
            : [],
        );

      toast({
        title: "⚠️ Missing required inputs",
        description: `Please provide: ${allMissingFields.map((k) => `"${k}"`).join(", ")}`,
        variant: "destructive",
      });
    },
    [
      missingInputs,
      scheduleName,
      toast,
      credentialsRequired,
      allCredentialsAreSet,
      missingCredentials,
    ],
  );

  // Action handlers
  const handleRun = useCallback(() => {
    if (!allRequiredInputsAreSet) {
      notifyMissingRequirements();
      return;
    }

    if (defaultRunType === "automatic-trigger") {
      // Setup trigger
      if (!scheduleName.trim()) {
        toast({
          title: "⚠️ Trigger name required",
          description: "Please provide a name for your trigger.",
          variant: "destructive",
        });
        return;
      }

      setupTriggerMutation.mutate({
        data: {
          name: presetName || scheduleName,
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
    scheduleName,
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

  const handleSchedule = useCallback(() => {
    if (!allRequiredInputsAreSet) {
      notifyMissingRequirements(true);
      return;
    }

    if (!scheduleName.trim()) {
      toast({
        title: "⚠️ Schedule name required",
        description: "Please provide a name for your schedule.",
        variant: "destructive",
      });
      return;
    }

    createScheduleMutation.mutate({
      graphId: agent.graph_id,
      data: {
        name: presetName || scheduleName,
        cron: cronExpression,
        inputs: inputValues,
        graph_version: agent.graph_version,
        credentials: inputCredentials,
      },
    });
  }, [
    allRequiredInputsAreSet,
    scheduleName,
    cronExpression,
    inputValues,
    inputCredentials,
    agent,
    notifyMissingRequirements,
    createScheduleMutation,
    toast,
  ]);

  function handleShowSchedule() {
    // Initialize with sensible defaults when entering schedule view
    setScheduleName((prev) => prev || defaultScheduleName);
    setCronExpression((prev) => prev || "0 9 * * 1");
    setShowScheduleView(true);
  }

  function handleGoBack() {
    setShowScheduleView(false);
    // Reset schedule fields on exit
    setScheduleName(defaultScheduleName);
    setCronExpression("0 9 * * 1");
  }

  function handleSetScheduleName(name: string) {
    setScheduleName(name);
  }

  function handleSetCronExpression(expression: string) {
    setCronExpression(expression);
  }

  const hasInputFields = useMemo(() => {
    return Object.keys(agentInputFields).length > 0;
  }, [agentInputFields]);

  return {
    isOpen,
    setIsOpen,
    showScheduleView,
    defaultRunType,
    inputValues,
    setInputValues,
    inputCredentials,
    setInputCredentials,
    presetName,
    presetDescription,
    setPresetName,
    setPresetDescription,
    scheduleName,
    cronExpression,
    allRequiredInputsAreSet,
    missingInputs,
    // Expose credential readiness for any UI hints if needed
    // but enforcement is already applied in allRequiredInputsAreSet
    // allCredentialsAreSet,
    // missingCredentials,
    agentInputFields,
    agentCredentialsInputFields,
    hasInputFields,
    isExecuting: executeGraphMutation.isPending,
    isCreatingSchedule: createScheduleMutation.isPending,
    isSettingUpTrigger: setupTriggerMutation.isPending,
    handleRun,
    handleSchedule,
    handleShowSchedule,
    handleGoBack,
    handleSetScheduleName,
    handleSetCronExpression,
  };
}
