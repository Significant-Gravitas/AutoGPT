import { Input } from "@/components/atoms/Input/Input";
import { MultiToggle } from "@/components/molecules/MultiToggle/MultiToggle";
import { Text } from "@/components/atoms/Text/Text";
import { Select } from "@/components/atoms/Select/Select";
import { useScheduleView } from "./useScheduleView";
import { useCallback, useState } from "react";
import { validateSchedule } from "./helpers";

interface Props {
  scheduleName: string;
  cronExpression: string;
  onScheduleNameChange: (name: string) => void;
  onCronExpressionChange: (expression: string) => void;
  onValidityChange?: (valid: boolean) => void;
}

export function ScheduleView({
  scheduleName,
  cronExpression: _cronExpression,
  onScheduleNameChange,
  onCronExpressionChange,
  onValidityChange,
}: Props) {
  const {
    repeat,
    selectedDays,
    time,
    repeatOptions,
    dayItems,
    setSelectedDays,
    handleRepeatChange,
    handleTimeChange,
    handleSelectAll,
    handleWeekdays,
    handleWeekends,
  } = useScheduleView({ onCronExpressionChange });

  function handleScheduleNameChange(e: React.ChangeEvent<HTMLInputElement>) {
    onScheduleNameChange(e.target.value);
  }

  const [errors, setErrors] = useState<{
    scheduleName?: string;
    time?: string;
  }>({});

  const validateNow = useCallback(
    (partial: { scheduleName?: string; time?: string }) => {
      const fieldErrors = validateSchedule({
        scheduleName,
        time,
        ...partial,
      });
      setErrors(fieldErrors);
      if (onValidityChange)
        onValidityChange(Object.keys(fieldErrors).length === 0);
    },
    [scheduleName, time, onValidityChange],
  );

  return (
    <div className="mt-6">
      <Input
        id="schedule-name"
        label="Schedule Name"
        value={scheduleName}
        onChange={(e) => {
          handleScheduleNameChange(e);
          validateNow({ scheduleName: e.target.value });
        }}
        placeholder="Enter a name for this schedule"
        error={errors.scheduleName}
      />

      <Select
        id="repeat"
        label="Repeats"
        value={repeat}
        onValueChange={handleRepeatChange}
        options={repeatOptions}
      />

      {repeat === "weekly" && (
        <div className="mb-8 space-y-3">
          <Text variant="body-medium" as="span" className="text-black">
            Repeats on
          </Text>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              className="h-[2.25rem] rounded-full border border-zinc-700 px-4 py-2 text-sm font-medium leading-[16px] text-black hover:bg-zinc-100"
              onClick={handleSelectAll}
            >
              Select all
            </button>
            <button
              type="button"
              className="h-[2.25rem] rounded-full border border-zinc-700 px-4 py-2 text-sm font-medium leading-[16px] text-black hover:bg-zinc-100"
              onClick={handleWeekdays}
            >
              Weekdays
            </button>
            <button
              type="button"
              className="h-[2.25rem] rounded-full border border-zinc-700 px-4 py-2 text-sm font-medium leading-[16px] text-black hover:bg-zinc-100"
              onClick={handleWeekends}
            >
              Weekends
            </button>
          </div>

          <MultiToggle
            items={dayItems}
            selectedValues={selectedDays}
            onChange={setSelectedDays}
            aria-label="Select days of week"
          />
        </div>
      )}

      <Input
        id="schedule-time"
        label="At"
        value={time}
        onChange={(e) => {
          const value = e.target.value.trim();
          handleTimeChange({ ...e, target: { ...e.target, value } } as any);
          validateNow({ time: value });
        }}
        placeholder="00:00"
        error={errors.time}
      />

      {/** Agent inputs are rendered in the main modal; none here. */}
    </div>
  );
}
