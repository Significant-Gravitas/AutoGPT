import { z } from "zod/v4";

export function validTimeZone(value: string) {
  try {
    new Intl.DateTimeFormat("en", { timeZone: value }).format(0);
    return value.length > 0;
  } catch {
    return false;
  }
}

export function localInstants(local: string, timeZone: string) {
  const [date, time] = local.split("T");
  if (
    !z.iso.date().safeParse(date).success ||
    !/^(?:[01]\d|2[0-3]):[0-5]\d$/.test(time ?? "") ||
    !validTimeZone(timeZone)
  )
    return [];
  const nominal = Date.parse(`${local}:00Z`);
  const format = new Intl.DateTimeFormat("en-CA-u-ca-gregory-nu-latn", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  });
  const result: string[] = [];
  for (let offset = -14 * 60; offset <= 14 * 60; offset += 15) {
    const instant = nominal + offset * 60_000;
    const parts = Object.fromEntries(
      format.formatToParts(instant).map((part) => [part.type, part.value]),
    );
    if (
      `${parts.year}-${parts.month}-${parts.day}T${parts.hour}:${parts.minute}` ===
      local
    )
      result.push(new Date(instant).toISOString());
  }
  return result;
}
