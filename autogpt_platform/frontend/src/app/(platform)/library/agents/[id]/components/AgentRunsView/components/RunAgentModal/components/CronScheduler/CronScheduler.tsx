"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { makeCronExpression } from "@/lib/cron-expression-utils";
import { FrequencySelect } from "./FrequencySelect";
import { WeeklyPicker } from "./WeeklyPicker";
import { MonthlyPicker } from "./MonthlyPicker";
import { YearlyPicker } from "./YearlyPicker";
import { CustomInterval } from "./CustomInterval";
import { TimeAt } from "./TimeAt";

export type CronFrequency =
  | "hourly"
  | "daily"
  | "weekly"
  | "monthly"
  | "yearly"
  | "custom"
  | "every minute";

type Props = {
  onCronExpressionChange: (cron: string) => void;
  initialCronExpression?: string;
};

export function CronScheduler({
  onCronExpressionChange,
  initialCronExpression,
}: Props) {
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

  // Parse provided cron only once to avoid feedback loops
  const parsedOnceRef = useRef(false);

  useEffect(() => {
    if (parsedOnceRef.current) return;
    parsedOnceRef.current = true;

    if (!initialCronExpression) {
      setFrequency("daily");
      setSelectedWeekDays([]);
      setSelectedMonthDays([]);
      setSelectedMonths([]);
      return;
    }

    const parts = initialCronExpression.trim().split(/\s+/);
    if (parts.length !== 5) return;

    const [minute, hour, dayOfMonth, month, dayOfWeek] = parts;
    setSelectedWeekDays([]);
    setSelectedMonthDays([]);
    setSelectedMonths([]);

    if (
      minute === "*" &&
      hour === "*" &&
      dayOfMonth === "*" &&
      month === "*" &&
      dayOfWeek === "*"
    ) {
      setFrequency("every minute");
      return;
    }

    if (
      hour === "*" &&
      dayOfMonth === "*" &&
      month === "*" &&
      dayOfWeek === "*" &&
      !minute.includes("/")
    ) {
      setFrequency("hourly");
      setSelectedMinute(minute);
      return;
    }

    if (
      minute.startsWith("*/") &&
      hour === "*" &&
      dayOfMonth === "*" &&
      month === "*" &&
      dayOfWeek === "*"
    ) {
      setFrequency("custom");
      const interval = parseInt(minute.substring(2));
      if (!isNaN(interval))
        setCustomInterval({ value: interval, unit: "minutes" });
      return;
    }

    if (
      hour.startsWith("*/") &&
      minute === "0" &&
      dayOfMonth === "*" &&
      month === "*" &&
      dayOfWeek === "*"
    ) {
      setFrequency("custom");
      const interval = parseInt(hour.substring(2));
      if (!isNaN(interval))
        setCustomInterval({ value: interval, unit: "hours" });
      return;
    }

    if (
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
        if (!isNaN(hourNum) && !isNaN(minuteNum))
          setSelectedTime(
            `${hourNum.toString().padStart(2, "0")}:${minuteNum.toString().padStart(2, "0")}`,
          );
      }
      return;
    }

    if (dayOfMonth === "*" && month === "*" && dayOfWeek === "*") {
      setFrequency("daily");
      const hourNum = parseInt(hour);
      const minuteNum = parseInt(minute);
      if (!isNaN(hourNum) && !isNaN(minuteNum))
        setSelectedTime(
          `${hourNum.toString().padStart(2, "0")}:${minuteNum.toString().padStart(2, "0")}`,
        );
      return;
    }

    if (dayOfWeek !== "*" && dayOfMonth === "*" && month === "*") {
      setFrequency("weekly");
      const hourNum = parseInt(hour);
      const minuteNum = parseInt(minute);
      if (!isNaN(hourNum) && !isNaN(minuteNum))
        setSelectedTime(
          `${hourNum.toString().padStart(2, "0")}:${minuteNum.toString().padStart(2, "0")}`,
        );
      const days = dayOfWeek
        .split(",")
        .map((d) => parseInt(d))
        .filter((d) => !isNaN(d));
      setSelectedWeekDays(days);
      return;
    }

    if (dayOfMonth !== "*" && month === "*" && dayOfWeek === "*") {
      setFrequency("monthly");
      const hourNum = parseInt(hour);
      const minuteNum = parseInt(minute);
      if (!isNaN(hourNum) && !isNaN(minuteNum))
        setSelectedTime(
          `${hourNum.toString().padStart(2, "0")}:${minuteNum.toString().padStart(2, "0")}`,
        );
      const days = dayOfMonth
        .split(",")
        .map((d) => parseInt(d))
        .filter((d) => !isNaN(d) && d >= 1 && d <= 31);
      setSelectedMonthDays(days);
      return;
    }

    if (dayOfMonth !== "*" && month !== "*" && dayOfWeek === "*") {
      setFrequency("yearly");
      const hourNum = parseInt(hour);
      const minuteNum = parseInt(minute);
      if (!isNaN(hourNum) && !isNaN(minuteNum))
        setSelectedTime(
          `${hourNum.toString().padStart(2, "0")}:${minuteNum.toString().padStart(2, "0")}`,
        );
      const months = month
        .split(",")
        .map((m) => parseInt(m))
        .filter((m) => !isNaN(m) && m >= 1 && m <= 12);
      setSelectedMonths(months);
    }
  }, [initialCronExpression]);

  const cronExpression = useMemo(
    () =>
      makeCronExpression({
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
          frequency === "custom"
            ? customInterval
            : { unit: "minutes", value: 1 },
      }),
    [
      frequency,
      selectedMinute,
      selectedTime,
      selectedWeekDays,
      selectedMonthDays,
      selectedMonths,
      customInterval,
    ],
  );

  useEffect(() => {
    onCronExpressionChange(cronExpression);
  }, [cronExpression, onCronExpressionChange]);

  return (
    <div>
      <FrequencySelect
        value={frequency}
        onChange={setFrequency}
        selectedMinute={selectedMinute}
        onMinuteChange={setSelectedMinute}
      />

      {frequency === "custom" && (
        <CustomInterval value={customInterval} onChange={setCustomInterval} />
      )}

      {frequency === "weekly" && (
        <WeeklyPicker
          values={selectedWeekDays}
          onChange={setSelectedWeekDays}
        />
      )}

      {frequency === "monthly" && (
        <MonthlyPicker
          values={selectedMonthDays}
          onChange={setSelectedMonthDays}
        />
      )}

      {frequency === "yearly" && (
        <YearlyPicker values={selectedMonths} onChange={setSelectedMonths} />
      )}

      {frequency !== "hourly" && (
        <TimeAt value={selectedTime} onChange={setSelectedTime} />
      )}
    </div>
  );
}
