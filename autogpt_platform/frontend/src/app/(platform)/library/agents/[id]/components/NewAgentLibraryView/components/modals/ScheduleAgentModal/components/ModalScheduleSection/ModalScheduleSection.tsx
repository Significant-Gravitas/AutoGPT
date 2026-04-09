import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { useState } from "react";
import { parseCronToForm, validateSchedule } from "./helpers";
import { TimezoneNotice } from "../TimezoneNotice/TimezoneNotice";
import { CronScheduler } from "../CronScheduler/CronScheduler";

interface Props {
  scheduleName: string;
  cronExpression: string;
  recommendedScheduleCron?: string | null;
  onScheduleNameChange: (name: string) => void;
  onCronExpressionChange: (expression: string) => void;
  onValidityChange?: (valid: boolean) => void;
}

export function ModalScheduleSection({
  scheduleName,
  cronExpression: _cronExpression,
  recommendedScheduleCron,
  onScheduleNameChange,
  onCronExpressionChange,
  onValidityChange,
}: Props) {
  function handleScheduleNameChange(e: React.ChangeEvent<HTMLInputElement>) {
    onScheduleNameChange(e.target.value);
  }

  const [errors, setErrors] = useState<{
    scheduleName?: string;
  }>({});

  function validateNow(partial: {
    scheduleName?: string;
    cronExpression?: string;
  }) {
    const cronExpression =
      partial.cronExpression ??
      _cronExpression ??
      recommendedScheduleCron ??
      "";
    const fieldErrors = validateSchedule({
      scheduleName: partial.scheduleName ?? scheduleName,
      time: parseCronToForm(cronExpression)?.time ?? "",
    });
    setErrors(fieldErrors);
    if (onValidityChange)
      onValidityChange(Object.keys(fieldErrors).length === 0);
  }

  return (
    <div className="mt-6">
      <Input
        id="schedule-name"
        label="Schedule Name"
        value={scheduleName}
        size="small"
        onChange={(e) => {
          handleScheduleNameChange(e);
          validateNow({ scheduleName: e.target.value });
        }}
        placeholder="Enter a name for this schedule"
        error={errors.scheduleName}
        className="max-w-80"
      />

      {recommendedScheduleCron && (
        <div className="mb-4 rounded-md bg-blue-50 p-3">
          <Text variant="body" className="text-blue-800">
            💡 This agent has a recommended schedule that has been pre-filled
            below. You can modify it as needed.
          </Text>
        </div>
      )}

      <div className="mt-1">
        <CronScheduler
          onCronExpressionChange={(expression) => {
            onCronExpressionChange(expression);
            validateNow({ cronExpression: expression });
          }}
          initialCronExpression={
            _cronExpression ?? recommendedScheduleCron ?? undefined
          }
        />
      </div>

      <TimezoneNotice />
    </div>
  );
}
