import { defineComponent } from "@openuidev/lang-core";
import { z } from "zod/v4";

export const mapLocationSchema = z.object({
  name: z.string().min(1).max(120),
  latitude: z.number().min(-85.05112878).max(85.05112878),
  longitude: z.number().min(-180).max(180),
  detail: z.string().max(1200),
  category: z.string().max(80),
});

export type MapLocation = z.infer<typeof mapLocationSchema>;

export const Map = defineComponent({
  name: "Map",
  description:
    "An interactive geographic map with numbered markers, searchable places, details, zoom, and a button to discuss the selected place in this chat. Supply verified latitude/longitude from the user or tools; never guess coordinates. This does not geocode, calculate routes, or request the user's location.",
  props: z.object({
    title: z.string(),
    description: z.string(),
    locations: z.array(mapLocationSchema).min(1).max(50),
  }),
  component: null,
});

export const Timeline = defineComponent({
  name: "Timeline",
  description:
    "A chronological itinerary, milestone plan, or event history. Preserve supplied dates and time zones; label proposed times as a plan.",
  props: z.object({
    title: z.string(),
    items: z
      .array(
        z.object({
          time: z.string(),
          title: z.string(),
          detail: z.string(),
          status: z.enum(["done", "current", "planned"]),
        }),
      )
      .max(20),
  }),
  component: null,
});

export const TrendChart = defineComponent({
  name: "TrendChart",
  description:
    "An interactive line chart for ordered observations or a time series. Values may be negative. Include only supplied observations; never invent missing periods.",
  props: z.object({
    title: z.string(),
    description: z.string(),
    unit: z.string(),
    points: z.array(z.object({ label: z.string(), value: z.number() })).max(60),
  }),
  component: null,
});

export const DonutChart = defineComponent({
  name: "DonutChart",
  description:
    "A part-to-whole breakdown with a total, legend, values, and percentages. Use nonnegative amounts in the same unit; do not combine overlapping categories.",
  props: z.object({
    title: z.string(),
    description: z.string(),
    unit: z.string(),
    points: z
      .array(z.object({ label: z.string(), value: z.number().nonnegative() }))
      .max(12),
  }),
  component: null,
});
