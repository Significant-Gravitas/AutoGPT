import {
  itineraryInstant,
  planItinerary,
  type ItineraryDraft,
  type ItineraryProps,
} from "./itinerary";

function escapeText(text: string) {
  return text
    .replace(/\\/g, "\\\\")
    .replace(/\r?\n|\r/g, "\\n")
    .replace(/;/g, "\\;")
    .replace(/,/g, "\\,");
}
function fold(line: string) {
  let bytes = 0;
  let result = "";
  const encoder = new TextEncoder();
  for (const char of line) {
    const length = encoder.encode(char).length;
    if (bytes + length > 75) {
      result += "\r\n ";
      bytes = 1;
    }
    result += char;
    bytes += length;
  }
  return result;
}
function utc(value: string) {
  return value.replace(/[-:]/g, "").replace(/\.\d{3}/, "");
}

export function itineraryCalendar(
  props: ItineraryProps,
  draft: ItineraryDraft,
  now = new Date(),
  namespace = "",
) {
  const { result, error } = planItinerary(props, draft);
  if (!result?.feasible || error)
    throw new Error(
      error || "Resolve schedule conflicts before downloading the calendar.",
    );
  const lines = [
    "BEGIN:VCALENDAR",
    "VERSION:2.0",
    "PRODID:-//AutoGPT//Chat plan//EN",
    "CALSCALE:GREGORIAN",
  ];
  for (const allocation of result.allocations)
    for (const [index, slot] of allocation.slots.entries()) {
      const start = itineraryInstant(slot.start, draft);
      const end = itineraryInstant(slot.end, draft);
      if (!start || !end)
        throw new Error("Resolve ambiguous local times before exporting.");
      const task = draft.tasks.find((task) => task.id === allocation.task.id);
      const place = props.locations.find((place) => place.id === task?.placeId);
      lines.push(
        "BEGIN:VEVENT",
        `UID:${escapeText(namespace ? `${namespace}:${props.name}` : props.name)}-${escapeText(allocation.task.id)}-${index}@autogpt.local`,
        `DTSTAMP:${utc(now.toISOString())}`,
        `DTSTART:${utc(start)}`,
        `DTEND:${utc(end)}`,
        `SUMMARY:${escapeText(allocation.task.title)}`,
        `DESCRIPTION:${escapeText(`Proposed plan from AutoGPT. Time zone: ${draft.timeZone}. Not a booking.`)}`,
        "STATUS:TENTATIVE",
      );
      if (place)
        lines.push(
          `LOCATION:${escapeText(place.name)}`,
          `GEO:${place.latitude};${place.longitude}`,
        );
      lines.push("END:VEVENT");
    }
  return [...lines, "END:VCALENDAR", ""].map(fold).join("\r\n");
}
