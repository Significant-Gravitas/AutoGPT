import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { useScheduleRunForm } from "./useScheduleRunForm";
import { InputSection } from "../InputSection/InputSection";
import { CredentialsSection } from "../CredentialsSection/CredentialsSection";
import { ScheduleSection } from "./ScheduleSection";
import { GraphMeta } from "@/app/api/__generated__/models/graphMeta";

interface ScheduleRunFormProps {
  agent: GraphMeta;
  onClose: () => void;
}

export function ScheduleRunForm({ agent, onClose }: ScheduleRunFormProps) {
  const {
    inputValues,
    setInputValues,
    credentialValues,
    setCredentialValues,
    scheduleName,
    setScheduleName,
    cronExpression,
    setCronExpression,
    isCreating,
    canCreate,
    handleCreateSchedule,
    errors,
  } = useScheduleRunForm({ agent, onClose });

  return (
    <div className="space-y-6 p-6">
      <div className="space-y-4">
        <h3 className="text-lg font-medium text-neutral-800">Schedule Setup</h3>

        <div className="space-y-4">
          <Input
            id="schedule-name"
            label="Schedule Name"
            value={scheduleName}
            onChange={(e) => setScheduleName(e.target.value)}
            placeholder="Enter a name for this schedule"
            error={errors.scheduleName}
          />

          <ScheduleSection
            cronExpression={cronExpression}
            onChange={setCronExpression}
            error={errors.cronExpression}
          />
        </div>

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
          <Button variant="ghost" onClick={onClose} disabled={isCreating}>
            Cancel
          </Button>
          <Button
            variant="primary"
            onClick={handleCreateSchedule}
            loading={isCreating}
            disabled={!canCreate}
          >
            Create Schedule
          </Button>
        </div>
      </div>
    </div>
  );
}
