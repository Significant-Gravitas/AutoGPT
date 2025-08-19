import { useState, useMemo } from "react";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { validateInputs, validateCredentials } from "../../helpers";
import { CredentialsMetaInput } from "@/app/api/__generated__/models/credentialsMetaInput";
import { GraphMeta } from "@/app/api/__generated__/models/graphMeta";

interface UseScheduleRunFormProps {
  agent: GraphMeta;
  onClose: () => void;
}

export function useScheduleRunForm({
  agent,
  onClose,
}: UseScheduleRunFormProps) {
  const [inputValues, setInputValues] = useState<Record<string, any>>({});
  const [credentialValues, setCredentialValues] = useState<
    Record<string, CredentialsMetaInput>
  >({});
  const [scheduleName, setScheduleName] = useState(`${agent.name} Schedule`);
  const [cronExpression, setCronExpression] = useState("0 9 * * 1"); // Default: Every Monday at 9 AM
  const { toast } = useToast();

  // TODO: Replace with actual API call when imports are fixed
  const createScheduleMutation = {
    mutate: (params: any) => {
      console.log("Would create schedule with:", params);
      toast({
        title: "✅ Schedule created successfully (mock)",
        description: "Next run: Monday at 9:00 AM",
      });
      onClose();
    },
    isPending: false,
  };

  const errors = useMemo(() => {
    const inputErrors = validateInputs(agent.input_schema, inputValues);
    const credentialErrors = validateCredentials(
      agent.credentials_input_schema,
      credentialValues,
    );
    const scheduleErrors: Record<string, string> = {};

    if (!scheduleName.trim()) {
      scheduleErrors.scheduleName = "Schedule name is required";
    }

    if (!cronExpression.trim()) {
      scheduleErrors.cronExpression = "Schedule pattern is required";
    }

    return { ...inputErrors, ...credentialErrors, ...scheduleErrors };
  }, [agent, inputValues, credentialValues, scheduleName, cronExpression]);

  const canCreate = useMemo(() => {
    return Object.keys(errors).length === 0;
  }, [errors]);

  function handleCreateSchedule() {
    if (!canCreate) return;

    createScheduleMutation.mutate({
      graphId: agent.id,
      data: {
        name: scheduleName,
        cron: cronExpression,
        graph_version: agent.version,
        inputs: inputValues,
        credentials: credentialValues,
      },
    });
  }

  return {
    inputValues,
    setInputValues,
    credentialValues,
    setCredentialValues,
    scheduleName,
    setScheduleName,
    cronExpression,
    setCronExpression,
    isCreating: createScheduleMutation.isPending,
    canCreate,
    handleCreateSchedule,
    errors,
  };
}
