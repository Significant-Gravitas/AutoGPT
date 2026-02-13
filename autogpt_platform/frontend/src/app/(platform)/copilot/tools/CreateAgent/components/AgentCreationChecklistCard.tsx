import { CheckCircleIcon, CircleIcon } from "@phosphor-icons/react";

export interface AgentCreationChecklistStep {
  title: string;
  description: string;
  status: "pending" | "in_progress" | "completed";
}

interface Props {
  steps: AgentCreationChecklistStep[];
}

export function AgentCreationChecklistCard({ steps }: Props) {
  return (
    <div className="space-y-3">
      <p className="text-sm text-muted-foreground">
        Creating your custom agent...
      </p>
      <div className="space-y-2">
        {steps.map((step) => (
          <div key={step.title} className="flex items-start gap-3">
            <div className="mt-0.5 flex-shrink-0">
              {step.status === "completed" ? (
                <CheckCircleIcon
                  size={20}
                  weight="fill"
                  className="text-green-500"
                />
              ) : (
                <CircleIcon
                  size={20}
                  weight="regular"
                  className="text-neutral-400"
                />
              )}
            </div>
            <div className="flex-1 space-y-1">
              <div
                className={`text-sm font-medium ${
                  step.status === "completed"
                    ? "text-neutral-900 dark:text-neutral-100"
                    : "text-muted-foreground"
                }`}
              >
                {step.title}
              </div>
              <div className="text-xs text-muted-foreground">
                {step.description}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
