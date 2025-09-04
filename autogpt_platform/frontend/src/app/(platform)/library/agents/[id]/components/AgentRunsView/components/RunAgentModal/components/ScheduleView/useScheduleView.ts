import { useEffect, useMemo, useState } from "react";

interface UseScheduleViewOptions {
  onCronExpressionChange: (expression: string) => void;
}

export function useScheduleView({
  onCronExpressionChange,
}: UseScheduleViewOptions) {
  const repeatOptions = useMemo(
    () => [
      { value: "daily", label: "Daily" },
      { value: "weekly", label: "Weekly" },
    ],
    [],
  );

  const dayItems = useMemo(
    () => [
      { value: "0", label: "Su" },
      { value: "1", label: "Mo" },
      { value: "2", label: "Tu" },
      { value: "3", label: "We" },
      { value: "4", label: "Th" },
      { value: "5", label: "Fr" },
      { value: "6", label: "Sa" },
    ],
    [],
  );

  const [repeat, setRepeat] = useState<string>("weekly");
  const [selectedDays, setSelectedDays] = useState<string[]>([]);
  const [time, setTime] = useState<string>("00:00");

  function handleRepeatChange(value: string) {
    setRepeat(value);
  }

  function handleTimeChange(e: React.ChangeEvent<HTMLInputElement>) {
    setTime(e.target.value.trim());
  }

  function parseTimeToHM(value: string): { h: number; m: number } {
    const match = /^([01]?\d|2[0-3]):([0-5]\d)$/.exec(value || "");
    if (!match) return { h: 0, m: 0 };
    return { h: Number(match[1]), m: Number(match[2]) };
  }

  // Helpful default: when switching to Weekly with no days picked, preselect Monday
  useEffect(() => {
    if (repeat === "weekly" && selectedDays.length === 0)
      setSelectedDays(["1"]);
  }, [repeat, selectedDays]);

  // Build cron string any time repeat/days/time change
  useEffect(() => {
    const { h, m } = parseTimeToHM(time);
    const minute = String(m);
    const hour = String(h);
    if (repeat === "daily") {
      onCronExpressionChange(`${minute} ${hour} * * *`);
      return;
    }

    const dow = selectedDays.length ? selectedDays.join(",") : "*";
    onCronExpressionChange(`${minute} ${hour} * * ${dow}`);
  }, [repeat, selectedDays, time, onCronExpressionChange]);

  function handleSelectAll() {
    setSelectedDays(["0", "1", "2", "3", "4", "5", "6"]);
  }

  function handleWeekdays() {
    setSelectedDays(["1", "2", "3", "4", "5"]);
  }

  function handleWeekends() {
    setSelectedDays(["0", "6"]);
  }

  return {
    // state
    repeat,
    selectedDays,
    time,
    // derived/static
    repeatOptions,
    dayItems,
    // handlers
    setSelectedDays,
    handleRepeatChange,
    handleTimeChange,
    handleSelectAll,
    handleWeekdays,
    handleWeekends,
  };
}
