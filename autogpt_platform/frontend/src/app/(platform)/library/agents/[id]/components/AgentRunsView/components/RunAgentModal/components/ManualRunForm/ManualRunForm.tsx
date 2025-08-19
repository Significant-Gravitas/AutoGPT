import { Button } from "@/components/atoms/Button/Button";
import { useManualRunForm } from "./useManualRunForm";
import { InputSection } from "../InputSection/InputSection";
import { CredentialsSection } from "../CredentialsSection/CredentialsSection";
import { GraphMeta } from "@/app/api/__generated__/models/graphMeta";

interface ManualRunFormProps {
  agent: GraphMeta;
  onClose: () => void;
}

export function ManualRunForm({ agent, onClose }: ManualRunFormProps) {
  const {
    inputValues,
    setInputValues,
    credentialValues,
    setCredentialValues,
    isRunning,
    canRun,
    handleRun,
    errors,
  } = useManualRunForm({ agent, onClose });

  return (
    <div className="space-y-6 p-6">
      <div className="space-y-4">
        <h3 className="text-lg font-medium text-neutral-800">Agent Setup</h3>

        <InputSection
          agent={agent}
          values={inputValues}
          onChange={setInputValues}
          errors={errors}
        />

        <CredentialsSection
          agent={agent}
          values={credentialValues}
          onChange={setCredentialValues}
          errors={errors}
        />
      </div>

      <div className="sticky bottom-0 border-t border-neutral-200 bg-white pt-4">
        <div className="flex justify-end gap-3">
          <Button variant="ghost" onClick={onClose} disabled={isRunning}>
            Cancel
          </Button>
          <Button
            variant="primary"
            onClick={handleRun}
            loading={isRunning}
            disabled={!canRun}
          >
            Run Agent
          </Button>
        </div>
      </div>
    </div>
  );
}
