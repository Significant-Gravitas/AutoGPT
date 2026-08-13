import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";

export function getTimeOfDayGreeting(hour = new Date().getHours()) {
  if (hour < 12) return "Good morning";
  if (hour < 18) return "Good afternoon";
  return "Good evening";
}

export function getHomeStatusLine(dashboard: HomeDashboardResponse): string {
  const attention = dashboard.attention.length;
  const active = dashboard.active_tasks.length;
  if (attention > 0 && active > 0) {
    return `${attention} ${attention === 1 ? "decision" : "decisions"} waiting · ${active} ${active === 1 ? "agent is" : "agents are"} working now`;
  }
  if (attention > 0) {
    return `${attention} ${attention === 1 ? "decision needs" : "decisions need"} you`;
  }
  if (active > 0) {
    return `All clear · ${active} ${active === 1 ? "agent is" : "agents are"} working now`;
  }
  return "Nothing needs you right now";
}

export function formatHeaderDate(
  value: Date,
  timeZone: string,
  locale?: string,
) {
  const date = new Date(value);
  const options = { timeZone };

  return {
    weekday: new Intl.DateTimeFormat(locale, {
      ...options,
      weekday: "long",
    }).format(date),
    calendarDate: new Intl.DateTimeFormat(locale, {
      ...options,
      month: "long",
      day: "numeric",
    }).format(date),
  };
}

export function formatDuration(totalSeconds: number): string {
  const totalMinutes = Math.round(totalSeconds / 60);
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  if (hours === 0) return `${minutes}m`;
  if (minutes === 0) return `${hours}h`;
  return `${hours}h ${minutes}m`;
}

export function formatCurrency(cents: number): string {
  return new Intl.NumberFormat(undefined, {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
  }).format(cents / 100);
}
