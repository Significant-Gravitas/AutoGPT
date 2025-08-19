import { Input } from "@/components/atoms/Input/Input";
import { Button } from "@/components/atoms/Button/Button";
import { useState } from "react";

interface ScheduleSectionProps {
  cronExpression: string;
  onChange: (cron: string) => void;
  error?: string;
}

const SCHEDULE_PRESETS = [
  { label: "Every hour", cron: "0 * * * *" },
  { label: "Daily at 9 AM", cron: "0 9 * * *" },
  { label: "Every Monday at 9 AM", cron: "0 9 * * 1" },
  { label: "Every weekday at 9 AM", cron: "0 9 * * 1-5" },
  { label: "Every month on 1st at 9 AM", cron: "0 9 1 * *" },
];

export function ScheduleSection({
  cronExpression,
  onChange,
  error,
}: ScheduleSectionProps) {
  const [showCustom, setShowCustom] = useState(
    !SCHEDULE_PRESETS.some((preset) => preset.cron === cronExpression),
  );

  function handlePresetSelect(cron: string) {
    onChange(cron);
    setShowCustom(false);
  }

  return (
    <div className="space-y-4">
      <div>
        <label className="mb-2 block text-sm font-medium text-neutral-700">
          Schedule Pattern
        </label>

        {!showCustom && (
          <div className="space-y-2">
            <div className="grid grid-cols-1 gap-2">
              {SCHEDULE_PRESETS.map((preset) => (
                <button
                  key={preset.cron}
                  type="button"
                  onClick={() => handlePresetSelect(preset.cron)}
                  className={`rounded-lg border p-3 text-left transition-colors ${
                    cronExpression === preset.cron
                      ? "border-blue-500 bg-blue-50 text-blue-700"
                      : "border-neutral-200 hover:border-neutral-300 hover:bg-neutral-50"
                  }`}
                >
                  <div className="font-medium">{preset.label}</div>
                  <div className="mt-1 font-mono text-xs text-neutral-500">
                    {preset.cron}
                  </div>
                </button>
              ))}
            </div>

            <Button
              type="button"
              variant="ghost"
              size="small"
              onClick={() => setShowCustom(true)}
              className="w-full"
            >
              Use custom cron expression
            </Button>
          </div>
        )}

        {showCustom && (
          <div className="space-y-2">
            <Input
              id="cron-expression"
              label="Custom Cron Expression"
              value={cronExpression}
              onChange={(e) => onChange(e.target.value)}
              placeholder="0 9 * * 1"
              error={error}
              hideLabel
            />
            <div className="text-xs text-neutral-500">
              Format: minute hour day month weekday (e.g., &quot;0 9 * * 1&quot;
              for every Monday at 9 AM)
            </div>
            <Button
              type="button"
              variant="ghost"
              size="small"
              onClick={() => setShowCustom(false)}
              className="w-full"
            >
              Use preset schedules
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
