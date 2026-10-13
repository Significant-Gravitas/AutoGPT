import { defineComponent } from "@openuidev/lang-core";
import { z } from "zod/v4";
import { identifier, shortText } from "./data-values";

export const localDateTime = z.object({
  local: z.string().max(16),
  timeZone: z.string().max(100),
  occurrence: z.enum(["", "earlier", "later"]),
});
export const DateTimeField = defineComponent({
  name: "DateTimeField",
  description:
    "Date/time and IANA time zone input inside Form. value has local as YYYY-MM-DDTHH:mm or blank, timeZone like America/Los_Angeles, and occurrence as blank/earlier/later. Local validation rejects nonexistent daylight-saving times and requires an explicit occurrence for ambiguous times. Publishes the local choice and resolved UTC instant; no booking or scheduling occurs. Use for appointments and deadlines; DateRangeField is for calendar dates without a time.",
  props: z.object({
    name: identifier,
    label: shortText,
    value: localDateTime,
    required: z.boolean(),
  }),
  component: null,
});
