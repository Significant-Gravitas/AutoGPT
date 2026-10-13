import { defineComponent } from "@openuidev/lang-core";
import { z } from "zod/v4";
import { identifier, shortText } from "./data-values";

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
    "An interactive geographic map with one to 50 numbered markers, searchable places, details, zoom, and a button to discuss the selected place in this chat. Supply verified latitude/longitude from the user or tools; never guess coordinates. This does not geocode, calculate routes, or request the user's location.",
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
    "A chronological itinerary, milestone plan, or event history with at most 20 items. Preserve supplied dates and time zones; label proposed times as a plan.",
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

const chartSeries = z.object({ key: identifier, label: shortText });

export const SeriesChart = defineComponent({
  name: "SeriesChart",
  description:
    "Line, bar or stacked chart for one or more series, with legend toggles, an accessible data table, and true null gaps. Use it for any chart, including a single series. kind is line/bar/stacked. All series use the same stated unit; stacked requires nonnegative values. Points preserve supplied order; each contains label and values keyed by series key, with null for a missing observation. Never connect across a missing period by dropping its point. Up to five series and 100 points.",
  props: z
    .object({
      title: shortText,
      description: z.string().max(1000),
      unit: z.string().max(50),
      kind: z.enum(["line", "bar", "stacked"]),
      series: z.array(chartSeries).min(1).max(5),
      points: z
        .array(
          z.object({
            label: shortText,
            values: z.record(identifier, z.number().nullable()),
          }),
        )
        .max(100),
    })
    .superRefine(({ series, points, kind }, context) => {
      const keys = series.map((item) => item.key);
      if (new Set(keys).size !== keys.length)
        context.addIssue({
          code: "custom",
          message: "Series keys must be unique.",
        });
      for (const point of points) {
        if (
          Object.keys(point.values).some((key) => !keys.includes(key)) ||
          keys.some((key) => !(key in point.values))
        )
          context.addIssue({
            code: "custom",
            message:
              "Each point must include every series key, using null for unknown observations.",
          });
        if (
          kind === "stacked" &&
          Object.values(point.values).some(
            (value) => value != null && value < 0,
          )
        )
          context.addIssue({
            code: "custom",
            message: "Use grouped bars or lines for signed values.",
          });
      }
    }),
  component: null,
});
