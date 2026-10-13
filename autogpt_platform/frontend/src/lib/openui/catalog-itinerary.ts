import { defineComponent } from "@openuidev/lang-core";
import { z } from "zod/v4";
import { Schedule, scheduleTaskSchema } from "./catalog-schedule";
import { mapLocationSchema } from "./catalog-sections";
import { identifier, sourceLink } from "./data-values";
import { validTimeZone } from "./temporal";

export const itineraryDay = Schedule.props.shape.days.element.extend({
  date: z.iso.date(),
});
export const itineraryTask = scheduleTaskSchema.extend({
  placeId: identifier.optional(),
});
export const itineraryLocation = mapLocationSchema.extend({ id: identifier });
export const travelLeg = z.object({
  from: identifier,
  to: identifier,
  minutes: z.number().int().min(1).max(1440),
  mode: z.string().min(1).max(80),
  validOn: z.iso.date(),
  source: sourceLink.optional(),
});
export const Itinerary = defineComponent({
  name: "Itinerary",
  description:
    "Connected calendar, editable availability, visit durations/order/completion and map inside Form. days add ISO date to Schedule's label/windows; timeZone is IANA. tasks add optional placeId referencing locations (verified coordinates with stable id). legs are supplied travel estimates with from/to place ids, minutes, mode, validOn date and optional source. Required transfers are inserted automatically between consecutive places; do NOT duplicate transfer tasks. Reordering or changing dates invalidates missing/stale legs and pauses the schedule until researched in chat. Completed visits remain visible on the map. Includes current-plan calendar download; this does not book or create calendar events externally. Up to14 days,12 visits,50 places,50 directed legs. Use Schedule for non-geographic work, Itinerary for trips with known dates/locations/transfers. Missing facts require research or clarification, never guess.",
  props: z
    .object({
      ...Schedule.props.shape,
      name: identifier,
      days: z.array(itineraryDay).min(1).max(14),
      tasks: z.array(itineraryTask).min(1).max(12),
      timeZone: z
        .string()
        .max(100)
        .refine(validTimeZone, "Use a valid IANA time zone."),
      locations: z.array(itineraryLocation).min(1).max(50),
      legs: z.array(travelLeg).max(50),
    })
    .superRefine((props, context) => {
      const schedule = Schedule.props.safeParse(props);
      if (!schedule.success)
        for (const issue of schedule.error.issues)
          context.addIssue({ code: "custom", message: issue.message });
      if (
        props.days.some(
          (day, index) => index > 0 && day.date <= props.days[index - 1].date,
        )
      )
        context.addIssue({
          code: "custom",
          message: "Use distinct chronological dates.",
        });
      const ids = props.locations.map((place) => place.id);
      if (
        new Set(ids).size !== ids.length ||
        props.tasks.some((task) => task.placeId && !ids.includes(task.placeId))
      )
        context.addIssue({
          code: "custom",
          message:
            "Use unique location ids and existing task place references.",
        });
      if (
        new Set(props.legs.map((leg) => `${leg.from}/${leg.to}/${leg.validOn}`))
          .size !== props.legs.length ||
        props.legs.some(
          (leg) =>
            leg.from === leg.to ||
            !ids.includes(leg.from) ||
            !ids.includes(leg.to),
        )
      )
        context.addIssue({
          code: "custom",
          message:
            "Use one directed travel estimate per date between existing, different places.",
        });
    }),
  component: null,
});
