import { useState, useMemo } from "react";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { validateInputs, validateCredentials } from "../../helpers";
import { GraphMeta } from "@/app/api/__generated__/models/graphMeta";
import { CredentialsMetaInput } from "@/app/api/__generated__/models/credentialsMetaInput";

interface UseManualRunFormProps {
  agent: GraphMeta;
  onClose: () => void;
}

export function useManualRunForm({ agent, onClose }: UseManualRunFormProps) {
  const [inputValues, setInputValues] = useState<Record<string, any>>({});
  const [credentialValues, setCredentialValues] = useState<
    Record<string, CredentialsMetaInput>
  >({});
  const { toast } = useToast();

  // TODO: Replace with actual API call when imports are fixed
  const executeAgentMutation = {
    mutate: (params: any) => {
      console.log("Would execute agent with:", params);
      toast({
        title: "✅ Agent started successfully (mock)",
        description: "Execution ID: mock-exec-id",
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
    return { ...inputErrors, ...credentialErrors };
  }, [agent, inputValues, credentialValues]);

  const canRun = useMemo(() => {
    return Object.keys(errors).length === 0;
  }, [errors]);

  function handleRun() {
    if (!canRun) return;

    executeAgentMutation.mutate({
      graphId: agent.id,
      graphVersion: agent.version,
      data: {
        inputs: inputValues,
        credentials_inputs: credentialValues,
      },
    });
  }

  return {
    inputValues,
    setInputValues,
    credentialValues,
    setCredentialValues,
    isRunning: executeAgentMutation.isPending,
    canRun,
    handleRun,
    errors,
  };
}
