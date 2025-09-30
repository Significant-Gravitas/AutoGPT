export type CronFrequency =
  | "every minute"
  | "hourly"
  | "daily"
  | "weekly"
  | "monthly"
  | "yearly"
  | "custom";

export type CronExpressionParams =
  | { frequency: "every minute" }
  | {
      frequency: "hourly";
      minute: number;
    }
  | ((
      | {
          frequency: "daily";
        }
      | {
          frequency: "weekly";
          /** 0-based list of weekdays: 0 = Monday ... 6 = Sunday */
          days: number[];
        }
      | {
          frequency: "monthly";
          /** 1-based list of month days */
          days: number[];
        }
      | {
          frequency: "yearly";
          /** 1-based list of months (1-12) */
          months: number[];
        }
      | {
          frequency: "custom";
          customInterval: { unit: string; value: number };
        }
    ) & {
      minute: number;
      hour: number;
    });

export function makeCronExpression(params: CronExpressionParams): string {
  const frequency = params.frequency;

  if (frequency === "every minute") return "* * * * *";
  if (frequency === "hourly") return `${params.minute} * * * *`;
  if (frequency === "daily") return `${params.minute} ${params.hour} * * *`;
  if (frequency === "weekly") {
    const { minute, hour, days } = params;
    if (days.length === 0) return ""; // Return empty string for invalid weekly schedule
    const weekDaysExpr = days.sort((a, b) => a - b).join(",");
    return `${minute} ${hour} * * ${weekDaysExpr}`;
  }
  if (frequency === "monthly") {
    const { minute, hour, days } = params;
    if (days.length === 0) return ""; // Return empty string for invalid monthly schedule
    const monthDaysExpr = days.sort((a, b) => a - b).join(",");
    return `${minute} ${hour} ${monthDaysExpr} * *`;
  }
  if (frequency === "yearly") {
    const { minute, hour, months } = params;
    if (months.length === 0) return ""; // Return empty string for invalid yearly schedule
    const monthList = months.sort((a, b) => a - b).join(",");
    return `${minute} ${hour} 1 ${monthList} *`;
  }
  if (frequency === "custom") {
    const { minute, hour, customInterval } = params;
    if (customInterval.unit === "minutes") {
      return `*/${customInterval.value} * * * *`;
    } else if (customInterval.unit === "hours") {
      return `0 */${customInterval.value} * * *`;
    } else {
      return `${minute} ${hour} */${customInterval.value} * *`;
    }
  }

  return "";
}

export function humanizeCronExpression(cronExpression: string): string {
  const parts = cronExpression.trim().split(/\s+/);
  if (parts.length !== 5) {
    throw new Error("Invalid cron expression format.");
  }

  const [minute, hour, dayOfMonth, month, dayOfWeek] = parts;

  // Handle every minute
  if (cronExpression === "* * * * *") {
    return "Every minute";
  }

  // Handle minute intervals (e.g., */5 * * * *)
  if (
    minute.startsWith("*/") &&
    hour === "*" &&
    dayOfMonth === "*" &&
    month === "*" &&
    dayOfWeek === "*"
  ) {
    const interval = minute.substring(2);
    return `Every ${interval} minutes`;
  }

  // Handle hour intervals (e.g., 30 * * * *)
  if (
    hour === "*" &&
    !minute.includes("/") &&
    dayOfMonth === "*" &&
    month === "*" &&
    dayOfWeek === "*"
  ) {
    return `Every hour at minute ${minute}`;
  }

  // Handle every N hours (e.g., 0 */2 * * *)
  if (
    hour.startsWith("*/") &&
    minute === "0" &&
    dayOfMonth === "*" &&
    month === "*" &&
    dayOfWeek === "*"
  ) {
    const interval = hour.substring(2);
    return `Every ${interval} hours`;
  }

  // Handle daily (e.g., 30 14 * * *)
  if (
    dayOfMonth === "*" &&
    month === "*" &&
    dayOfWeek === "*" &&
    !minute.includes("/") &&
    !hour.includes("/")
  ) {
    return `Every day at ${formatTime(hour, minute)}`;
  }

  // Handle weekly (e.g., 30 14 * * 1,3,5)
  if (
    dayOfWeek !== "*" &&
    dayOfMonth === "*" &&
    month === "*" &&
    !minute.includes("/") &&
    !hour.includes("/")
  ) {
    const days = getDayNames(dayOfWeek);
    return `Every ${days} at ${formatTime(hour, minute)}`;
  }

  // Handle monthly (e.g., 30 14 1,15 * *)
  if (
    dayOfMonth !== "*" &&
    month === "*" &&
    dayOfWeek === "*" &&
    !minute.includes("/") &&
    !hour.includes("/")
  ) {
    const days = dayOfMonth.split(",").map(Number);
    const dayList = days.join(", ");
    return `On day ${dayList} of every month at ${formatTime(hour, minute)}`;
  }

  // Handle yearly (e.g., 30 14 1 1,6,12 *)
  if (
    dayOfMonth !== "*" &&
    month !== "*" &&
    dayOfWeek === "*" &&
    !minute.includes("/") &&
    !hour.includes("/")
  ) {
    const months = getMonthNames(month);
    return `Every year on the 1st day of ${months} at ${formatTime(hour, minute)}`;
  }

  // Handle custom minute intervals with other fields as * (e.g., every N minutes)
  if (
    minute.includes("/") &&
    hour === "*" &&
    dayOfMonth === "*" &&
    month === "*" &&
    dayOfWeek === "*"
  ) {
    const interval = minute.split("/")[1];
    return `Every ${interval} minutes`;
  }

  // Handle custom hour intervals with other fields as * (e.g., every N hours)
  if (
    hour.includes("/") &&
    minute === "0" &&
    dayOfMonth === "*" &&
    month === "*" &&
    dayOfWeek === "*"
  ) {
    const interval = hour.split("/")[1];
    return `Every ${interval} hours`;
  }

  // Handle specific days with custom intervals (e.g., every N days)
  if (
    dayOfMonth.startsWith("*/") &&
    month === "*" &&
    dayOfWeek === "*" &&
    !minute.includes("/") &&
    !hour.includes("/")
  ) {
    const interval = dayOfMonth.substring(2);
    return `Every ${interval} days at ${formatTime(hour, minute)}`;
  }

  return `Cron Expression: ${cronExpression}`;
}

function formatTime(hour: string, minute: string): string {
  // Cron expressions are now stored in the schedule's timezone (not UTC)
  // So we just format the time as-is without conversion
  const formattedHour = padZero(hour);
  const formattedMinute = padZero(minute);
  return `${formattedHour}:${formattedMinute}`;
}

function padZero(value: string): string {
  return value.padStart(2, "0");
}

function getDayNames(dayOfWeek: string): string {
  const days = dayOfWeek.split(",").map(Number);
  const dayNames = days
    .map((d) => {
      const names = [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
      ];
      return names[d] || `Unknown(${d})`;
    })
    .join(", ");
  return dayNames;
}

function getMonthNames(month: string): string {
  const months = month.split(",").map(Number);
  const monthNames = months
    .map((m) => {
      const names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
      ];
      return names[m - 1] || `Unknown(${m})`;
    })
    .join(", ");
  return monthNames;
}
