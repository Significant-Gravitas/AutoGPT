export class CronExpressionManager {
  generateCronExpression(
    frequency: string,
    selectedTime: string,
    selectedDays: number[],
    selectedMinute: string,
    customInterval: { unit: string; value: number },
  ): string {
    const [hours, minutes] = selectedTime.split(":").map(Number);
    let expression = "";

    switch (frequency) {
      case "minute":
        expression = "* * * * *";
        break;
      case "hour":
        expression = `${selectedMinute} * * * *`;
        break;
      case "daily":
        expression = `${minutes} ${hours} * * *`;
        break;
      case "weekly":
        const days = selectedDays.join(",");
        expression = `${minutes} ${hours} * * ${days}`;
        break;
      case "monthly":
        const monthDays = selectedDays.sort((a, b) => a - b).join(",");
        expression = `${minutes} ${hours} ${monthDays} * *`;
        break;
      case "yearly":
        const monthList = selectedDays
          .map((d) => d + 1)
          .sort((a, b) => a - b)
          .join(",");
        expression = `${minutes} ${hours} 1 ${monthList} *`;
        break;
      case "custom":
        if (customInterval.unit === "minutes") {
          expression = `*/${customInterval.value} * * * *`;
        } else if (customInterval.unit === "hours") {
          expression = `0 */${customInterval.value} * * *`;
        } else {
          expression = `${minutes} ${hours} */${customInterval.value} * *`;
        }
        break;
      default:
        expression = "";
    }
    return expression;
  }

  generateDescription(cronExpression: string): string {
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
      return `Every day at ${this.formatTime(hour, minute)}`;
    }

    // Handle weekly (e.g., 30 14 * * 1,3,5)
    if (
      dayOfWeek !== "*" &&
      dayOfMonth === "*" &&
      month === "*" &&
      !minute.includes("/") &&
      !hour.includes("/")
    ) {
      const days = this.getDayNames(dayOfWeek);
      return `Every ${days} at ${this.formatTime(hour, minute)}`;
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
      return `On day ${dayList} of every month at ${this.formatTime(hour, minute)}`;
    }

    // Handle yearly (e.g., 30 14 1 1,6,12 *)
    if (
      dayOfMonth !== "*" &&
      month !== "*" &&
      dayOfWeek === "*" &&
      !minute.includes("/") &&
      !hour.includes("/")
    ) {
      const months = this.getMonthNames(month);
      return `Every year on the 1st day of ${months} at ${this.formatTime(hour, minute)}`;
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
      return `Every ${interval} days at ${this.formatTime(hour, minute)}`;
    }

    return `Cron Expression: ${cronExpression}`;
  }

  private formatTime(hour: string, minute: string): string {
    const formattedHour = this.padZero(hour);
    const formattedMinute = this.padZero(minute);
    return `${formattedHour}:${formattedMinute}`;
  }

  private padZero(value: string): string {
    return value.padStart(2, "0");
  }

  private getDayNames(dayOfWeek: string): string {
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

  private getMonthNames(month: string): string {
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
}
