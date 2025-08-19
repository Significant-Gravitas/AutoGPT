import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { useAutomaticTriggerForm } from "./useAutomaticTriggerForm";
import { InputSection } from "../InputSection/InputSection";
import { CredentialsSection } from "../CredentialsSection/CredentialsSection";
import { GraphMeta } from "@/app/api/__generated__/models/graphMeta";

interface AutomaticTriggerFormProps {
  agent: GraphMeta;
  onClose: () => void;
}

export function AutomaticTriggerForm({
  agent,
  onClose,
}: AutomaticTriggerFormProps) {
  const {
    inputValues,
    setInputValues,
    credentialValues,
    setCredentialValues,
    triggerName,
    setTriggerName,
    triggerDescription,
    setTriggerDescription,
    isCreating,
    canCreate,
    handleCreateTrigger,
    errors,
  } = useAutomaticTriggerForm({ agent, onClose });

  return (
    <div className="space-y-6 p-6">
      <div className="space-y-4">
        <h3 className="text-lg font-medium text-neutral-800">
          Automatic Trigger Setup
        </h3>

        <div className="rounded-lg border border-blue-200 bg-blue-50 p-4">
          <div className="flex items-start">
            <div className="flex-shrink-0">
              <svg
                className="h-5 w-5 text-blue-400"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  fillRule="evenodd"
                  d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
                  clipRule="evenodd"
                />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-blue-800">
                Webhook Trigger
              </h3>
              <div className="mt-2 text-sm text-blue-700">
                <p>
                  This will create a webhook endpoint that automatically runs
                  your agent when triggered by external events.
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <Input
            id="trigger-name"
            label="Trigger Name"
            value={triggerName}
            onChange={(e) => setTriggerName(e.target.value)}
            placeholder="Enter a name for this trigger"
            error={errors.triggerName}
          />

          <Input
            id="trigger-description"
            label="Description"
            type="textarea"
            rows={3}
            value={triggerDescription}
            onChange={(e) => setTriggerDescription(e.target.value)}
            placeholder="Describe what this trigger does"
            error={errors.triggerDescription}
          />
        </div>

        <InputSection
          agent={agent}
          values={inputValues}
          onChange={setInputValues}
          errors={errors}
          title="Trigger Configuration"
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
          <Button variant="ghost" onClick={onClose} disabled={isCreating}>
            Cancel
          </Button>
          <Button
            variant="primary"
            onClick={handleCreateTrigger}
            loading={isCreating}
            disabled={!canCreate}
          >
            Set up Trigger
          </Button>
        </div>
      </div>
    </div>
  );
}
