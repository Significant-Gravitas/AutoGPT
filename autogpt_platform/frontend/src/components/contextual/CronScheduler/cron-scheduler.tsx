import React, { useEffect, useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Select } from "@/components/atoms/Select/Select";
import { Text } from "@/components/atoms/Text/Text";
import { TimeInput } from "@/components/atoms/TimeInput/TimeInput";
import { CronFrequency, makeCronExpression } from "@/lib/cron-expression-utils";

const weekDays = [
  { label: "Su", value: 0 },
  { label: "Mo", value: 1 },
  { label: "Tu", value: 2 },
  { label: "We", value: 3 },
  { label: "Th", value: 4 },
  { label: "Fr", value: 5 },
  { label: "Sa", value: 6 },
];

const frequencyOptions = [
  { value: "hourly", label: "Every Hour" },
  { value: "daily", label: "Daily" },
  { value: "weekly", label: "Weekly" },
  { value: "monthly", label: "Monthly" },
  { value: "yearly", label: "Yearly" },
  { value: "custom", label: "Custom" },
];

const minuteOptions = [0, 15, 30, 45].map((min) => ({
  value: min.toString(),
  label: min.toString(),
}));

const intervalUnitOptions = [
  { value: "minutes", label: "Minutes" },
  { value: "hours", label: "Hours" },
  { value: "days", label: "Days" },
];

type IntervalUnit = "minutes" | "hours" | "days";

const months = [
  { label: "Jan", value: "January" },
  { label: "Feb", value: "February" },
  { label: "Mar", value: "March" },
  { label: "Apr", value: "April" },
  { label: "May", value: "May" },
  { label: "Jun", value: "June" },
  { label: "Jul", value: "July" },
  { label: "Aug", value: "August" },
  { label: "Sep", value: "September" },
  { label: "Oct", value: "October" },
  { label: "Nov", value: "November" },
  { label: "Dec", value: "December" },
];

type CronSchedulerProps = {
  onCronExpressionChange: (cronExpression: string) => void;
  initialCronExpression?: string;
};

export function CronScheduler({
  onCronExpressionChange,
  initialCronExpression,
}: CronSchedulerProps): React.ReactElement {
  const [frequency, setFrequency] = useState<CronFrequency>("daily");
  const [selectedMinute, setSelectedMinute] = useState<string>("0");
  const [selectedTime, setSelectedTime] = useState<string>("09:00");
  const [selectedWeekDays, setSelectedWeekDays] = useState<number[]>([]);
  const [selectedMonthDays, setSelectedMonthDays] = useState<number[]>([]);
  const [selectedMonths, setSelectedMonths] = useState<number[]>([]);
  const [customInterval, setCustomInterval] = useState<{
    value: number;
    unit: IntervalUnit;
  }>({ value: 1, unit: "minutes" });

  // Parse initial cron expression and set state
  useEffect(() => {
    if (!initialCronExpression) {
      // Reset to defaults when no initial expression
      setFrequency("daily");
      setSelectedWeekDays([]);
      setSelectedMonthDays([]);
      setSelectedMonths([]);
      return;
    }

    const parts = initialCronExpression.trim().split(/\s+/);
    if (parts.length !== 5) return;

    const [minute, hour, dayOfMonth, month, dayOfWeek] = parts;

    // Reset all state first
    setSelectedWeekDays([]);
    setSelectedMonthDays([]);
    setSelectedMonths([]);

    // Parse patterns in order of specificity
    if (
      minute === "*" &&
      hour === "*" &&
      dayOfMonth === "*" &&
      month === "*" &&
      dayOfWeek === "*"
    ) {
      setFrequency("every minute");
    } else if (
      hour === "*" &&
      dayOfMonth === "*" &&
      month === "*" &&
      dayOfWeek === "*" &&
      !minute.includes("/")
    ) {
      setFrequency("hourly");
      setSelectedMinute(minute);
    } else if (
      minute.startsWith("*/") &&
      hour === "*" &&
      dayOfMonth === "*" &&
      month === "*" &&
      dayOfWeek === "*"
    ) {
      setFrequency("custom");
      const interval = parseInt(minute.substring(2));
      if (!isNaN(interval)) {
        setCustomInterval({ value: interval, unit: "minutes" });
      }
    } else if (
      hour.startsWith("*/") &&
      minute === "0" &&
      dayOfMonth === "*" &&
      month === "*" &&
      dayOfWeek === "*"
    ) {
      setFrequency("custom");
      const interval = parseInt(hour.substring(2));
      if (!isNaN(interval)) {
        setCustomInterval({ value: interval, unit: "hours" });
      }
    } else if (
      dayOfMonth.startsWith("*/") &&
      month === "*" &&
      dayOfWeek === "*" &&
      !minute.includes("/") &&
      !hour.includes("/")
    ) {
      setFrequency("custom");
      const interval = parseInt(dayOfMonth.substring(2));
      if (!isNaN(interval)) {
        setCustomInterval({ value: interval, unit: "days" });
        const hourNum = parseInt(hour);
        const minuteNum = parseInt(minute);
        if (!isNaN(hourNum) && !isNaN(minuteNum)) {
          setSelectedTime(
            `${hourNum.toString().padStart(2, "0")}:${minuteNum.toString().padStart(2, "0")}`,
          );
        }
      }
    } else if (dayOfMonth === "*" && month === "*" && dayOfWeek === "*") {
      setFrequency("daily");
      const hourNum = parseInt(hour);
      const minuteNum = parseInt(minute);
      if (!isNaN(hourNum) && !isNaN(minuteNum)) {
        setSelectedTime(
          `${hourNum.toString().padStart(2, "0")}:${minuteNum.toString().padStart(2, "0")}`,
        );
      }
    } else if (dayOfWeek !== "*" && dayOfMonth === "*" && month === "*") {
      setFrequency("weekly");
      const hourNum = parseInt(hour);
      const minuteNum = parseInt(minute);
      if (!isNaN(hourNum) && !isNaN(minuteNum)) {
        setSelectedTime(
          `${hourNum.toString().padStart(2, "0")}:${minuteNum.toString().padStart(2, "0")}`,
        );
      }
      const days = dayOfWeek
        .split(",")
        .map((d) => parseInt(d))
        .filter((d) => !isNaN(d));
      setSelectedWeekDays(days);
    } else if (dayOfMonth !== "*" && month === "*" && dayOfWeek === "*") {
      setFrequency("monthly");
      const hourNum = parseInt(hour);
      const minuteNum = parseInt(minute);
      if (!isNaN(hourNum) && !isNaN(minuteNum)) {
        setSelectedTime(
          `${hourNum.toString().padStart(2, "0")}:${minuteNum.toString().padStart(2, "0")}`,
        );
      }
      const days = dayOfMonth
        .split(",")
        .map((d) => parseInt(d))
        .filter((d) => !isNaN(d) && d >= 1 && d <= 31);
      setSelectedMonthDays(days);
    } else if (dayOfMonth !== "*" && month !== "*" && dayOfWeek === "*") {
      setFrequency("yearly");
      const hourNum = parseInt(hour);
      const minuteNum = parseInt(minute);
      if (!isNaN(hourNum) && !isNaN(minuteNum)) {
        setSelectedTime(
          `${hourNum.toString().padStart(2, "0")}:${minuteNum.toString().padStart(2, "0")}`,
        );
      }
      const months = month
        .split(",")
        .map((m) => parseInt(m))
        .filter((m) => !isNaN(m) && m >= 1 && m <= 12);
      setSelectedMonths(months);
    }
  }, [initialCronExpression]);

  useEffect(() => {
    const cronExpr = makeCronExpression({
      frequency,
      minute:
        frequency === "hourly"
          ? parseInt(selectedMinute)
          : parseInt(selectedTime.split(":")[1]),
      hour: parseInt(selectedTime.split(":")[0]),
      days:
        frequency === "weekly"
          ? selectedWeekDays
          : frequency === "monthly"
            ? selectedMonthDays
            : [],
      months: frequency === "yearly" ? selectedMonths : [],
      customInterval:
        frequency === "custom" ? customInterval : { unit: "minutes", value: 1 },
    });
    onCronExpressionChange(cronExpr);
  }, [
    frequency,
    selectedMinute,
    selectedTime,
    selectedWeekDays,
    selectedMonthDays,
    selectedMonths,
    customInterval,
    onCronExpressionChange,
  ]);

  return (
    <div className="max-w-md space-y-6">
      <div className="space-y-4">
        <Text variant="large-medium" as="span">
          Repeat
        </Text>

        <Select
          id="cron-frequency"
          label="Repeat"
          hideLabel
          size="md"
          wrapperClassName="mb-0"
          placeholder="Select frequency"
          value={frequency}
          onValueChange={(value) => setFrequency(value as CronFrequency)}
          options={frequencyOptions}
        />

        {frequency === "hourly" && (
          <div className="flex items-center gap-2">
            <Text variant="body-medium" as="span">
              At minute
            </Text>
            <Select
              id="cron-minute"
              label="At minute"
              hideLabel
              size="md"
              className="w-24"
              wrapperClassName="mb-0"
              placeholder="Select minute"
              value={selectedMinute}
              onValueChange={setSelectedMinute}
              options={minuteOptions}
            />
          </div>
        )}

        {frequency === "custom" && (
          <div className="flex items-center gap-2">
            <Text variant="body-medium" as="span">
              Every
            </Text>
            <Input
              id="cron-custom-interval"
              label="Every"
              hideLabel
              type="number"
              size="md"
              min="1"
              className="w-20"
              wrapperClassName="mb-0 w-20"
              value={customInterval.value}
              onChange={(e) =>
                setCustomInterval({
                  ...customInterval,
                  value: parseInt(e.target.value),
                })
              }
            />
            <Select
              id="cron-custom-interval-unit"
              label="Interval unit"
              hideLabel
              size="md"
              className="w-32"
              wrapperClassName="mb-0"
              value={customInterval.unit}
              onValueChange={(value) =>
                setCustomInterval({
                  ...customInterval,
                  unit: value as IntervalUnit,
                })
              }
              options={intervalUnitOptions}
            />
          </div>
        )}
      </div>

      {frequency === "weekly" && (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Text variant="body-medium" as="span">
              On
            </Text>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => {
                if (selectedWeekDays.length === weekDays.length) {
                  setSelectedWeekDays([]);
                } else {
                  setSelectedWeekDays(weekDays.map((day) => day.value));
                }
              }}
            >
              {selectedWeekDays.length === weekDays.length
                ? "Deselect All"
                : "Select All"}
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setSelectedWeekDays([1, 2, 3, 4, 5])}
            >
              Weekdays
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setSelectedWeekDays([0, 6])}
            >
              Weekends
            </Button>
          </div>
          <div className="flex flex-wrap gap-2">
            {weekDays.map((day) => (
              <Button
                key={day.value}
                variant={
                  selectedWeekDays.includes(day.value) ? "primary" : "secondary"
                }
                size="md"
                className="h-10 w-10 min-w-0 p-0"
                onClick={() => {
                  setSelectedWeekDays((prev) =>
                    prev.includes(day.value)
                      ? prev.filter((d) => d !== day.value)
                      : [...prev, day.value],
                  );
                }}
              >
                {day.label}
              </Button>
            ))}
          </div>
          {selectedWeekDays.length === 0 && (
            <Text variant="body" tone="danger">
              Please select at least one day of the week
            </Text>
          )}
        </div>
      )}
      {frequency === "monthly" && (
        <div className="space-y-4">
          <Text variant="body-medium" as="span">
            Days of Month
          </Text>
          <div className="flex gap-2">
            <Button
              variant={
                selectedMonthDays.length === 31 ? "primary" : "secondary"
              }
              size="md"
              onClick={() => {
                setSelectedMonthDays(
                  Array.from({ length: 31 }, (_, i) => i + 1),
                );
              }}
            >
              All Days
            </Button>
            <Button
              variant={
                selectedMonthDays.length < 31 && selectedMonthDays.length > 0
                  ? "primary"
                  : "secondary"
              }
              size="md"
              onClick={() => {
                setSelectedMonthDays([]);
              }}
            >
              Customize
            </Button>
            <Button
              variant="secondary"
              size="md"
              onClick={() => setSelectedMonthDays([15])}
            >
              15th
            </Button>
            <Button
              variant="secondary"
              size="md"
              onClick={() => setSelectedMonthDays([31])}
            >
              Last Day
            </Button>
          </div>
          {selectedMonthDays.length < 31 && (
            <div className="flex flex-wrap gap-2">
              {Array.from({ length: 31 }, (_, i) => (
                <Button
                  key={i + 1}
                  variant={
                    selectedMonthDays.includes(i + 1) ? "primary" : "secondary"
                  }
                  size="md"
                  className="h-10 w-10 min-w-0 p-0"
                  onClick={() => {
                    setSelectedMonthDays((prev) =>
                      prev.includes(i + 1)
                        ? prev.filter((d) => d !== i + 1)
                        : [...prev, i + 1],
                    );
                  }}
                >
                  {i + 1}
                </Button>
              ))}
            </div>
          )}
          {selectedMonthDays.length === 0 && (
            <Text variant="body" tone="danger">
              Please select at least one day of the month
            </Text>
          )}
        </div>
      )}
      {frequency === "yearly" && (
        <div className="space-y-4">
          <Text variant="body-medium" as="span">
            Months
          </Text>
          <div className="flex gap-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => {
                if (selectedMonths.length === months.length) {
                  setSelectedMonths([]);
                } else {
                  setSelectedMonths(
                    Array.from({ length: 12 }, (_, i) => i + 1),
                  );
                }
              }}
            >
              {selectedMonths.length === months.length
                ? "Deselect All"
                : "Select All"}
            </Button>
          </div>
          <div className="flex flex-wrap gap-2">
            {months.map((month, i) => {
              const monthNumber = i + 1;
              return (
                <Button
                  key={i}
                  variant={
                    selectedMonths.includes(monthNumber)
                      ? "primary"
                      : "secondary"
                  }
                  size="md"
                  className="min-w-0 px-2 py-1"
                  onClick={() => {
                    setSelectedMonths((prev) =>
                      prev.includes(monthNumber)
                        ? prev.filter((m) => m !== monthNumber)
                        : [...prev, monthNumber],
                    );
                  }}
                >
                  {month.label}
                </Button>
              );
            })}
          </div>
          {selectedMonths.length === 0 && (
            <Text variant="body" tone="danger">
              Please select at least one month
            </Text>
          )}
        </div>
      )}

      {frequency !== "hourly" &&
        !(frequency === "custom" && customInterval.unit !== "days") && (
          <div className="flex items-center gap-4 space-y-2">
            <Text variant="body-medium" as="span" className="pt-2">
              At
            </Text>
            <TimeInput
              value={selectedTime}
              onChange={setSelectedTime}
              aria-label="Time"
              size="md"
              wrapperClassName="mb-0"
            />
          </div>
        )}
    </div>
  );
}
