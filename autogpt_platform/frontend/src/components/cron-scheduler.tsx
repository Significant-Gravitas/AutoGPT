import React, { useEffect, useState } from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
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
};

export function CronScheduler({
  onCronExpressionChange,
}: CronSchedulerProps): React.ReactElement {
  const [frequency, setFrequency] = useState<CronFrequency>("daily");
  const [selectedMinute, setSelectedMinute] = useState<string>("0");
  const [selectedTime, setSelectedTime] = useState<string>("09:00");
  const [selectedWeekDays, setSelectedWeekDays] = useState<number[]>([]);
  const [selectedMonthDays, setSelectedMonthDays] = useState<number[]>([]);
  const [selectedMonths, setSelectedMonths] = useState<number[]>([]);
  const [customInterval, setCustomInterval] = useState<{
    value: number;
    unit: "minutes" | "hours" | "days";
  }>({ value: 1, unit: "minutes" });
  const [showCustomDays, setShowCustomDays] = useState<boolean>(false);

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
        <Label className="text-base font-medium">Repeat</Label>

        <Select
          value={frequency}
          onValueChange={(value: CronFrequency) => setFrequency(value)}
        >
          <SelectTrigger>
            <SelectValue placeholder="Select frequency" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="every minute">Every Minute</SelectItem>
            <SelectItem value="hourly">Every Hour</SelectItem>
            <SelectItem value="daily">Daily</SelectItem>
            <SelectItem value="weekly">Weekly</SelectItem>
            <SelectItem value="monthly">Monthly</SelectItem>
            <SelectItem value="yearly">Yearly</SelectItem>
            <SelectItem value="custom">Custom</SelectItem>
          </SelectContent>
        </Select>

        {frequency === "hourly" && (
          <div className="flex items-center gap-2">
            <Label>At minute</Label>
            <Select value={selectedMinute} onValueChange={setSelectedMinute}>
              <SelectTrigger className="w-24">
                <SelectValue placeholder="Select minute" />
              </SelectTrigger>
              <SelectContent>
                {[0, 15, 30, 45].map((min) => (
                  <SelectItem key={min} value={min.toString()}>
                    {min}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}

        {frequency === "custom" && (
          <div className="flex items-center gap-2">
            <Label>Every</Label>
            <Input
              type="number"
              min="1"
              className="w-20"
              value={customInterval.value}
              onChange={(e) =>
                setCustomInterval({
                  ...customInterval,
                  value: parseInt(e.target.value),
                })
              }
            />
            <Select
              value={customInterval.unit}
              onValueChange={(value: any) =>
                setCustomInterval({ ...customInterval, unit: value })
              }
            >
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="minutes">Minutes</SelectItem>
                <SelectItem value="hours">Hours</SelectItem>
                <SelectItem value="days">Days</SelectItem>
              </SelectContent>
            </Select>
          </div>
        )}
      </div>

      {frequency === "weekly" && (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Label>On</Label>
            <Button
              variant="outline"
              className="h-8 px-2 py-1 text-xs"
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
              variant="outline"
              className="h-8 px-2 py-1 text-xs"
              onClick={() => setSelectedWeekDays([1, 2, 3, 4, 5])}
            >
              Weekdays
            </Button>
            <Button
              variant="outline"
              className="h-8 px-2 py-1 text-xs"
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
                  selectedWeekDays.includes(day.value) ? "default" : "outline"
                }
                className="h-10 w-10 p-0"
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
        </div>
      )}
      {frequency === "monthly" && (
        <div className="space-y-4">
          <Label>Days of Month</Label>
          <div className="flex gap-2">
            <Button
              variant={!showCustomDays ? "default" : "outline"}
              onClick={() => {
                setShowCustomDays(false);
                setSelectedMonthDays(
                  Array.from({ length: 31 }, (_, i) => i + 1),
                );
              }}
            >
              All Days
            </Button>
            <Button
              variant={showCustomDays ? "default" : "outline"}
              onClick={() => {
                setShowCustomDays(true);
                setSelectedMonthDays([]);
              }}
            >
              Customize
            </Button>
            <Button
              variant="outline"
              onClick={() => setSelectedMonthDays([15])}
            >
              15th
            </Button>
            <Button
              variant="outline"
              onClick={() => setSelectedMonthDays([31])}
            >
              Last Day
            </Button>
          </div>
          {showCustomDays && (
            <div className="flex flex-wrap gap-2">
              {Array.from({ length: 31 }, (_, i) => (
                <Button
                  key={i + 1}
                  variant={
                    selectedMonthDays.includes(i + 1) ? "default" : "outline"
                  }
                  className="h-10 w-10 p-0"
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
        </div>
      )}
      {frequency === "yearly" && (
        <div className="space-y-4">
          <Label>Months</Label>
          <div className="flex gap-2">
            <Button
              variant="outline"
              className="h-8 px-2 py-1 text-xs"
              onClick={() => {
                if (selectedMonths.length === months.length) {
                  setSelectedMonths([]);
                } else {
                  setSelectedMonths(Array.from({ length: 12 }, (_, i) => i));
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
                    selectedMonths.includes(monthNumber) ? "default" : "outline"
                  }
                  className="px-2 py-1"
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
        </div>
      )}

      {frequency !== "every minute" && frequency !== "hourly" && (
        <div className="flex items-center gap-4 space-y-2">
          <Label className="pt-2">At</Label>
          <Input
            type="time"
            value={selectedTime}
            onChange={(e) => setSelectedTime(e.target.value)}
          />
        </div>
      )}
    </div>
  );
}
