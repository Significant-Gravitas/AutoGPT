import { Input } from "@/components/atoms/Input/Input";
import { LibraryAgent } from "@/lib/autogpt-server-api/types";

interface Props {
  agent: LibraryAgent;
  inputValues: Record<string, any>;
  onInputChange: (key: string, value: string) => void;
  variant?: "default" | "schedule";
}

export function AgentInputFields({
  agent,
  inputValues,
  onInputChange,
  variant = "default",
}: Props) {
  const hasInputFields =
    agent.input_schema &&
    typeof agent.input_schema === "object" &&
    "properties" in agent.input_schema;

  if (!hasInputFields) {
    const emptyStateClass =
      variant === "schedule"
        ? "rounded-lg bg-neutral-50 p-4 text-sm text-neutral-500"
        : "p-4 text-sm text-neutral-500";

    return (
      <div className={emptyStateClass}>
        No input fields required for this agent
      </div>
    );
  }

  return (
    <>
      {Object.entries((agent.input_schema as any).properties || {}).map(
        ([key, schema]: [string, any]) => (
          <Input
            key={key}
            id={key}
            label={schema.title || key}
            value={inputValues[key] || ""}
            onChange={(e) => onInputChange(key, e.target.value)}
            placeholder={schema.description}
          />
        ),
      )}
    </>
  );
}
