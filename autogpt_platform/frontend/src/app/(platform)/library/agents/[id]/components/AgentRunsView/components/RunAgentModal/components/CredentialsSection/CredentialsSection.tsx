import { CredentialsMetaInput } from "@/app/api/__generated__/models/credentialsMetaInput";
import { GraphMeta } from "@/app/api/__generated__/models/graphMeta";
import { Button } from "@/components/atoms/Button/Button";

interface CredentialsSectionProps {
  agent: GraphMeta;
  values: Record<string, CredentialsMetaInput>;
  onChange: (values: Record<string, CredentialsMetaInput>) => void;
  errors: Record<string, string>;
}

export function CredentialsSection({
  agent,
  values,
  onChange,
  errors,
}: CredentialsSectionProps) {
  const credentialsSchema = agent.credentials_input_schema;
  const credentialFields = credentialsSchema?.properties || {};

  if (Object.keys(credentialFields).length === 0) {
    return null;
  }

  function handleCredentialChange(key: string, value?: CredentialsMetaInput) {
    const newValues = { ...values };
    if (value === undefined) {
      delete newValues[key];
    } else {
      newValues[key] = value;
    }
    onChange(newValues);
  }

  return (
    <div className="space-y-4">
      <h4 className="text-md font-medium text-neutral-800">Credentials</h4>

      <div className="space-y-4">
        {Object.entries(credentialFields).map(
          ([key, inputSubSchema]: [string, any]) => (
            <div key={key} className="space-y-2">
              <label className="block text-sm font-medium text-neutral-700">
                {inputSubSchema.title || key} Credentials
              </label>

              <div className="rounded-lg border border-neutral-200 bg-neutral-50 p-4">
                <div className="flex items-center justify-between">
                  <div className="text-sm text-neutral-600">
                    {values[key]
                      ? `Connected: ${values[key].title || values[key].provider}`
                      : "No credentials selected"}
                  </div>
                  <Button
                    variant="ghost"
                    size="small"
                    onClick={() => {
                      // Mock credential selection
                      handleCredentialChange(key, {
                        id: `mock-${key}-id`,
                        provider: inputSubSchema.provider || key,
                        type: "api_key",
                        title: `Mock ${key} credentials`,
                      });
                    }}
                  >
                    {values[key] ? "Change" : "Connect"}
                  </Button>
                </div>
              </div>

              {errors[key] && (
                <p className="text-sm text-red-500">{errors[key]}</p>
              )}
            </div>
          ),
        )}
      </div>
    </div>
  );
}
