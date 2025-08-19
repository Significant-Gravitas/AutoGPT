import { useState, useMemo } from "react";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { validateInputs, validateCredentials } from "../../helpers";
import { GraphMeta } from "@/app/api/__generated__/models/graphMeta";
import { CredentialsMetaInput } from "@/app/api/__generated__/models/credentialsMetaInput";

interface UseAutomaticTriggerFormProps {
  agent: GraphMeta;
  onClose: () => void;
}

export function useAutomaticTriggerForm({
  agent,
  onClose,
}: UseAutomaticTriggerFormProps) {
  const [inputValues, setInputValues] = useState<Record<string, any>>({});
  const [credentialValues, setCredentialValues] = useState<
    Record<string, CredentialsMetaInput>
  >({});
  const [triggerName, setTriggerName] = useState(`${agent.name} Trigger`);
  const [triggerDescription, setTriggerDescription] = useState("");
  const { toast } = useToast();

  // TODO: Replace with actual API call when imports are fixed
  const setupTriggerMutation = {
    mutate: (params: any) => {
      console.log("Would setup trigger with:", params);
      toast({
        title: "✅ Trigger set up successfully (mock)",
        description: "Webhook ID: mock-webhook-id",
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
    const triggerErrors: Record<string, string> = {};

    if (!triggerName.trim()) {
      triggerErrors.triggerName = "Trigger name is required";
    }

    return { ...inputErrors, ...credentialErrors, ...triggerErrors };
  }, [agent, inputValues, credentialValues, triggerName]);

  const canCreate = useMemo(() => {
    return Object.keys(errors).length === 0;
  }, [errors]);

  function handleCreateTrigger() {
    if (!canCreate) return;

    setupTriggerMutation.mutate({
      data: {
        name: triggerName,
        description: triggerDescription,
        graph_id: agent.id,
        graph_version: agent.version,
        trigger_config: inputValues,
        agent_credentials: credentialValues,
      },
    });
  }

  return {
    inputValues,
    setInputValues,
    credentialValues,
    setCredentialValues,
    triggerName,
    setTriggerName,
    triggerDescription,
    setTriggerDescription,
    isCreating: setupTriggerMutation.isPending,
    canCreate,
    handleCreateTrigger,
    errors,
  };
}
