import { GraphMeta } from "@/app/api/__generated__/models/graphMeta";
import { Input } from "@/components/atoms/Input/Input";

interface InputSectionProps {
  agent: GraphMeta;
  values: Record<string, any>;
  onChange: (values: Record<string, any>) => void;
  errors: Record<string, string>;
  title?: string;
}

export function InputSection({
  agent,
  values,
  onChange,
  errors,
  title = "Agent Inputs",
}: InputSectionProps) {
  const inputSchema = agent.input_schema;
  const inputFields = Object.fromEntries(
    Object.entries(inputSchema.properties || {}).filter(
      ([_, subSchema]: [string, any]) => !subSchema.hidden,
    ),
  );

  if (Object.keys(inputFields).length === 0) {
    return null;
  }

  function handleInputChange(key: string, value: any) {
    onChange({
      ...values,
      [key]: value,
    });
  }

  return (
    <div className="space-y-4">
      <h4 className="text-md font-medium text-neutral-800">{title}</h4>

      <div className="space-y-4">
        {Object.entries(inputFields).map(
          ([key, inputSubSchema]: [string, any]) => (
            <div key={key} className="space-y-2">
              <Input
                id={`agent-input-${key}`}
                label={inputSubSchema.title || key}
                value={values[key] ?? inputSubSchema.default ?? ""}
                placeholder={inputSubSchema.description}
                onChange={(e) => handleInputChange(key, e.target.value)}
                error={errors[key]}
                type={inputSubSchema.type === "number" ? "number" : "text"}
              />
            </div>
          ),
        )}
      </div>
    </div>
  );
}
