import { Input } from "@/components/atoms/Input/Input";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { AgentInputFields } from "../AgentInputFields/AgentInputFields";

interface Props {
  agent: LibraryAgent;
  scheduleName: string;
  cronExpression: string;
  inputValues: Record<string, any>;
  onScheduleNameChange: (name: string) => void;
  onCronExpressionChange: (expression: string) => void;
  onInputChange: (key: string, value: string) => void;
}

export function ScheduleView({
  agent,
  scheduleName,
  cronExpression,
  inputValues,
  onScheduleNameChange,
  onCronExpressionChange,
  onInputChange,
}: Props) {
  return (
    <div className="space-y-4">
      <h3 className="text-lg font-medium text-neutral-800">Schedule Setup</h3>

      <Input
        id="schedule-name"
        label="Schedule Name"
        value={scheduleName}
        onChange={(e) => onScheduleNameChange(e.target.value)}
        placeholder="Enter a name for this schedule"
      />

      <Input
        id="cron-expression"
        label="Schedule Pattern"
        value={cronExpression}
        onChange={(e) => onCronExpressionChange(e.target.value)}
        placeholder="0 9 * * 1"
        hint={
          <span className="text-xs text-neutral-500">
            Format: minute hour day month weekday
          </span>
        }
      />

      <AgentInputFields
        agent={agent}
        inputValues={inputValues}
        onInputChange={onInputChange}
        variant="schedule"
      />
    </div>
  );
}
