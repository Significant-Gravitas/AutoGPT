import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { useManualTriggerForm } from "./useManualTriggerForm";
import { GraphMeta } from "@/app/api/__generated__/models/graphMeta";

interface ManualTriggerFormProps {
  agent: GraphMeta;
  onClose: () => void;
}

export function ManualTriggerForm({ agent, onClose }: ManualTriggerFormProps) {
  const { apiEndpoint, apiKey, isGenerating, handleGenerateEndpoint } =
    useManualTriggerForm({ agent, onClose });

  return (
    <div className="space-y-6 p-6">
      <div className="space-y-4">
        <h3 className="text-lg font-medium text-neutral-800">
          Manual Trigger Setup
        </h3>

        <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
          <div className="flex items-start">
            <div className="flex-shrink-0">
              <svg
                className="h-5 w-5 text-amber-400"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  fillRule="evenodd"
                  d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                  clipRule="evenodd"
                />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-amber-800">
                API Under Development
              </h3>
              <div className="mt-2 text-sm text-amber-700">
                <p>
                  The manual trigger API is currently being developed. This will
                  allow you to trigger the agent via HTTP requests.
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <div>
            <label className="mb-2 block text-sm font-medium text-neutral-700">
              API Endpoint (Preview)
            </label>
            <div className="relative">
              <Input
                id="api-endpoint"
                label="API Endpoint"
                value={apiEndpoint}
                readOnly
                hideLabel
                className="font-mono text-sm"
              />
              <Button
                type="button"
                variant="ghost"
                size="small"
                onClick={() => navigator.clipboard.writeText(apiEndpoint)}
                className="absolute right-2 top-1/2 -translate-y-1/2"
              >
                Copy
              </Button>
            </div>
          </div>

          <div>
            <label className="mb-2 block text-sm font-medium text-neutral-700">
              API Key (Preview)
            </label>
            <div className="relative">
              <Input
                id="api-key"
                label="API Key"
                value={apiKey}
                type="password"
                readOnly
                hideLabel
                className="font-mono text-sm"
              />
              <Button
                type="button"
                variant="ghost"
                size="small"
                onClick={() => navigator.clipboard.writeText(apiKey)}
                className="absolute right-2 top-1/2 -translate-y-1/2"
              >
                Copy
              </Button>
            </div>
          </div>

          <div className="rounded-lg border border-neutral-200 bg-neutral-50 p-4">
            <h4 className="mb-2 text-sm font-medium text-neutral-800">
              Example Usage (Coming Soon)
            </h4>
            <pre className="overflow-x-auto text-xs text-neutral-600">
              {`curl -X POST "${apiEndpoint}" \\
  -H "Authorization: Bearer ${apiKey}" \\
  -H "Content-Type: application/json" \\
  -d '{
    "inputs": {
      "param1": "value1"
    }
  }'`}
            </pre>
          </div>
        </div>
      </div>

      <div className="sticky bottom-0 border-t border-neutral-200 bg-white pt-4">
        <div className="flex justify-end gap-3">
          <Button variant="ghost" onClick={onClose}>
            Close
          </Button>
          <Button
            variant="primary"
            onClick={handleGenerateEndpoint}
            loading={isGenerating}
            disabled
          >
            Generate Endpoint (Coming Soon)
          </Button>
        </div>
      </div>
    </div>
  );
}
