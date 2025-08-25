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
  const [scheduleName, setScheduleName] = useState("");
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
            title: "✅ Agent execution started",
            description: "Your agent is now running.",
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
            title: "✅ Schedule created",
            description: `Agent scheduled to run: ${scheduleName}`,
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
            title: "✅ Trigger setup complete",
            description: "Your webhook trigger is now active.",
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

  // Validation logic
  const [allRequiredInputsAreSet, missingInputs] = useMemo(() => {
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

  const notifyMissingInputs = useCallback(
    (needScheduleName: boolean = false) => {
      const allMissingFields = (
        needScheduleName && !scheduleName ? ["schedule_name"] : []
      ).concat(missingInputs);

      toast({
        title: "⚠️ Missing required inputs",
        description: `Please provide: ${allMissingFields.map((k) => `"${k}"`).join(", ")}`,
        variant: "destructive",
      });
    },
    [missingInputs, scheduleName, toast],
  );

  // Action handlers
  const handleRun = useCallback(() => {
    if (!allRequiredInputsAreSet) {
      notifyMissingInputs();
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
          name: scheduleName,
          description: `Trigger for ${agent.name}`,
          graph_id: agent.graph_id,
          graph_version: agent.graph_version,
          trigger_config: inputValues,
          agent_credentials: {}, // TODO: Add credentials handling if needed
        },
      });
    } else {
      // Manual execution
      executeGraphMutation.mutate({
        graphId: agent.graph_id,
        graphVersion: agent.graph_version,
        data: {
          inputs: inputValues,
          credentials_inputs: {}, // TODO: Add credentials handling if needed
        },
      });
    }
  }, [
    allRequiredInputsAreSet,
    defaultRunType,
    scheduleName,
    inputValues,
    agent,
    notifyMissingInputs,
    setupTriggerMutation,
    executeGraphMutation,
    toast,
  ]);

  const handleSchedule = useCallback(() => {
    if (!allRequiredInputsAreSet) {
      notifyMissingInputs(true);
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
        name: scheduleName,
        cron: cronExpression,
        inputs: inputValues,
        graph_version: agent.graph_version,
        credentials: {}, // TODO: Add credentials handling if needed
      },
    });
  }, [
    allRequiredInputsAreSet,
    scheduleName,
    cronExpression,
    inputValues,
    agent,
    notifyMissingInputs,
    createScheduleMutation,
    toast,
  ]);

  function handleShowSchedule() {
    setShowScheduleView(true);
  }

  function handleGoBack() {
    setShowScheduleView(false);
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
    scheduleName,
    cronExpression,
    allRequiredInputsAreSet,
    missingInputs,
    agentInputFields,
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
