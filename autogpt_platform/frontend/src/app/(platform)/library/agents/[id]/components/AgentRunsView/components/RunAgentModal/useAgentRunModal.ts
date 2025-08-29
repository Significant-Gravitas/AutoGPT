import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useState, useCallback, useMemo } from "react";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { usePostV1ExecuteGraphAgent } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { usePostV1CreateExecutionSchedule as useCreateSchedule } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { usePostV2SetupTrigger } from "@/app/api/__generated__/endpoints/presets/presets";
import { ExecuteGraphResponse } from "@/app/api/__generated__/models/executeGraphResponse";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";
import {
  collectMissingFields,
  getErrorMessage,
  deriveReadiness,
  getVisibleInputFields,
  getCredentialFields,
} from "./helpers";

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
  const executeGraphMutation = usePostV1ExecuteGraphAgent();
  const createScheduleMutation = useCreateSchedule();
  const setupTriggerMutation = usePostV2SetupTrigger();

  // Input schema validation
  const agentInputSchema = useMemo(
    () => agent.input_schema || { properties: {}, required: [] },
    [agent.input_schema],
  );

  const agentInputFields = useMemo(
    () => getVisibleInputFields(agentInputSchema),
    [agentInputSchema],
  );

  const agentCredentialsInputFields = useMemo(
    () => getCredentialFields(agent.credentials_input_schema),
    [agent.credentials_input_schema],
  );

  // Validation logic (presence checks derived from schemas)
  const {
    missingInputs,
    missingCredentials,
    credentialsRequired,
    allRequiredInputsAreSet,
  } = useMemo(
    () =>
      deriveReadiness({
        inputSchema: agentInputSchema,
        credentialsProperties: agentCredentialsInputFields,
        values: inputValues,
        credentialsValues: inputCredentials,
      }),
    [
      agentInputSchema,
      agentCredentialsInputFields,
      inputValues,
      inputCredentials,
    ],
  );

  const notifyMissingRequirements = useCallback(
    (needScheduleName: boolean = false) => {
      const allMissingFields = collectMissingFields({
        needScheduleName,
        scheduleName,
        missingInputs,
        credentialsRequired,
        allCredentialsAreSet: missingCredentials.length === 0,
        missingCredentials,
      });

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
      missingCredentials,
    ],
  );

  function showError(title: string, error: unknown) {
    toast({
      title,
      description: getErrorMessage(error),
      variant: "destructive",
    });
  }

  async function handleRun() {
    if (!allRequiredInputsAreSet) {
      notifyMissingRequirements();
      return;
    }

    const shouldUseTrigger = defaultRunType === "automatic-trigger";

    if (shouldUseTrigger) {
      // Setup trigger
      const hasScheduleName = scheduleName.trim().length > 0;
      if (!hasScheduleName) {
        toast({
          title: "⚠️ Trigger name required",
          description: "Please provide a name for your trigger.",
          variant: "destructive",
        });
        return;
      }
      try {
        const nameToUse = presetName || scheduleName;
        const descriptionToUse =
          presetDescription || `Trigger for ${agent.name}`;
        const response = await setupTriggerMutation.mutateAsync({
          data: {
            name: nameToUse,
            description: descriptionToUse,
            graph_id: agent.graph_id,
            graph_version: agent.graph_version,
            trigger_config: inputValues,
            agent_credentials: inputCredentials,
          },
        });
        if (response.status === 200) {
          toast({ title: "Trigger setup complete" });
          callbacks?.onSetupTrigger?.(response.data);
          setIsOpen(false);
        } else {
          throw new Error(JSON.stringify(response?.data?.detail));
        }
      } catch (error: any) {
        showError("❌ Failed to setup trigger", error);
      }
    } else {
      // Manual execution
      try {
        const response = await executeGraphMutation.mutateAsync({
          graphId: agent.graph_id,
          graphVersion: agent.graph_version,
          data: {
            inputs: inputValues,
            credentials_inputs: inputCredentials,
          },
        });

        if (response.status === 200) {
          toast({ title: "Agent execution started" });
          callbacks?.onRun?.(response.data);
          setIsOpen(false);
        } else {
          throw new Error(JSON.stringify(response?.data?.detail));
        }
      } catch (error: any) {
        showError("Failed to execute agent", error);
      }
    }
  }

  async function handleSchedule() {
    if (!allRequiredInputsAreSet) {
      notifyMissingRequirements(true);
      return;
    }

    const hasScheduleName = scheduleName.trim().length > 0;
    if (!hasScheduleName) {
      toast({
        title: "⚠️ Schedule name required",
        description: "Please provide a name for your schedule.",
        variant: "destructive",
      });
      return;
    }
    try {
      const nameToUse = presetName || scheduleName;
      const response = await createScheduleMutation.mutateAsync({
        graphId: agent.graph_id,
        data: {
          name: nameToUse,
          cron: cronExpression,
          inputs: inputValues,
          graph_version: agent.graph_version,
          credentials: inputCredentials,
        },
      });
      if (response.status === 200) {
        toast({ title: "Schedule created" });
        callbacks?.onCreateSchedule?.(response.data);
        setIsOpen(false);
      }
    } catch (error: any) {
      showError("❌ Failed to create schedule", error);
    }
  }

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
