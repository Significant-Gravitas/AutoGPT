import { z } from "zod/v4";
import { Itinerary, itineraryDay, itineraryTask } from "./catalog-itinerary";
import { Schedule } from "./catalog-schedule";
import { durationError } from "./schedule";
import { scheduleTransfers } from "./itinerary-transfers";
import { localInstants, validTimeZone } from "./temporal";

export type ItineraryProps = z.infer<typeof Itinerary.props>;
export const itineraryDraft = z.object({
  days: z
    .array(
      itineraryDay.extend({
        date: z.string().max(10),
        windows: z
          .array(z.object({ start: z.string().max(5), end: z.string().max(5) }))
          .min(1)
          .max(4),
      }),
    )
    .min(1)
    .max(14),
  timeZone: z.string().max(100),
  tasks: z
    .array(
      itineraryTask.extend({
        minutes: z.union([z.number(), z.string().max(10)]),
      }),
    )
    .min(1)
    .max(12),
  selected: z.string(),
});
export type ItineraryDraft = z.infer<typeof itineraryDraft>;

export function planItinerary(props: ItineraryProps, draft: ItineraryDraft) {
  const invalid = itineraryInputError(draft);
  if (invalid) return { result: null, error: invalid };
  const { result, transfers, error } = scheduleTransfers(props, draft);
  if (error) return { result: null, error };
  if (result)
    for (const allocation of result.allocations) {
      for (const slot of allocation.slots) {
        const start = itineraryInstant(slot.start, draft);
        const end = itineraryInstant(slot.end, draft);
        if (
          !start ||
          !end ||
          Date.parse(end) - Date.parse(start) !==
            (slot.end - slot.start) * 60_000
        )
          return {
            result: null,
            error:
              "A slot crosses an ambiguous or skipped daylight-saving time. Adjust the available times before exporting.",
          };
      }
    }
  return { result, error: "", transfers };
}

export function itineraryInputError(draft: ItineraryDraft) {
  if (!validTimeZone(draft.timeZone)) return "Enter a valid IANA time zone.";
  if (
    draft.days.some(
      (day, index) =>
        !z.iso.date().safeParse(day.date).success ||
        (index > 0 && day.date <= draft.days[index - 1].date),
    )
  )
    return "Use valid, distinct dates in chronological order.";
  const windows = Schedule.props.safeParse({
    name: "plan",
    title: "Plan",
    days: draft.days,
    tasks: draft.tasks.map((task) => ({ ...task, minutes: 1 })),
  });
  if (!windows.success)
    return "Availability windows must be valid, ordered, nonoverlapping and start before they end.";
  if (draft.tasks.some((task) => !task.done && durationError(task.minutes)))
    return "Enter valid whole-minute visit durations.";
  return "";
}

export function itineraryInstant(minutes: number, draft: ItineraryDraft) {
  const day = draft.days[Math.floor(minutes / 1440)];
  if (!day) return null;
  const time = minutes % 1440;
  const matches = localInstants(
    `${day.date}T${String(Math.floor(time / 60)).padStart(2, "0")}:${String(time % 60).padStart(2, "0")}`,
    draft.timeZone,
  );
  return matches.length === 1 ? matches[0] : null;
}
