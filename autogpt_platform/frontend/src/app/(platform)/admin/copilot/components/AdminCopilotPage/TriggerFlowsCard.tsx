import { ChatSessionStartType } from "@/app/api/__generated__/models/chatSessionStartType";
import { Button } from "@/components/atoms/Button/Button";
import { Card } from "@/components/atoms/Card/Card";

const triggerOptions = [
  {
    label: "Trigger CTA",
    description:
      "Runs the beta invite CTA flow even if the user would not normally qualify.",
    startType: ChatSessionStartType.AUTOPILOT_INVITE_CTA,
    variant: "primary" as const,
  },
  {
    label: "Trigger Nightly",
    description:
      "Runs the nightly proactive Autopilot flow immediately for the selected user.",
    startType: ChatSessionStartType.AUTOPILOT_NIGHTLY,
    variant: "outline" as const,
  },
  {
    label: "Trigger Callback",
    description:
      "Runs the callback re-engagement flow without checking the normal callback cohort.",
    startType: ChatSessionStartType.AUTOPILOT_CALLBACK,
    variant: "secondary" as const,
  },
];

interface Props {
  disabled: boolean;
  isTriggeringSession: boolean;
  pendingTriggerType: ChatSessionStartType | null;
  onTriggerSession: (startType: ChatSessionStartType) => void;
}

export function TriggerFlowsCard({
  disabled,
  isTriggeringSession,
  pendingTriggerType,
  onTriggerSession,
}: Props) {
  return (
    <Card className="border border-zinc-200 shadow-sm">
      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-1">
          <h2 className="text-xl font-semibold text-zinc-900">Trigger flows</h2>
          <p className="text-sm text-zinc-600">
            Each action creates a new session immediately for the selected user.
          </p>
        </div>

        <div className="flex flex-col gap-3">
          {triggerOptions.map((option) => (
            <div
              key={option.startType}
              className="rounded-2xl border border-zinc-200 p-4"
            >
              <div className="flex flex-col gap-3">
                <div className="flex flex-col gap-1">
                  <span className="font-medium text-zinc-900">
                    {option.label}
                  </span>
                  <p className="text-sm text-zinc-600">{option.description}</p>
                </div>
                <Button
                  variant={option.variant}
                  disabled={disabled || isTriggeringSession}
                  loading={pendingTriggerType === option.startType}
                  onClick={() => onTriggerSession(option.startType)}
                >
                  {option.label}
                </Button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
}
