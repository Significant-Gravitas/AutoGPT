import type { ItineraryProps, ItineraryDraft } from "./itinerary";
import { calculateSchedule, type ScheduleDraft } from "./schedule";

type Transfer = ItineraryProps["legs"][number];
export function withTransfers(props: ItineraryProps, draft: ItineraryDraft) {
  const tasks: ScheduleDraft[] = [];
  const transfers = new Map<string, Transfer>();
  let lastPlace: string | undefined;
  let started = false;
  for (const task of draft.tasks) {
    if (task.done) {
      tasks.push(task);
      if (!started && task.placeId) lastPlace = task.placeId;
      continue;
    }
    started = true;
    if (task.placeId && lastPlace && lastPlace !== task.placeId) {
      const leg = props.legs.find(
        (leg) => leg.from === lastPlace && leg.to === task.placeId,
      );
      if (!leg)
        return {
          tasks,
          transfers,
          error:
            "This order needs a new travel estimate. Ask the assistant to research the missing transfer before using the times.",
        };
      let id = `transfer-${tasks.length}`;
      while (draft.tasks.some((item) => item.id === id)) id += "x";
      tasks.push({
        id,
        title: transferTitle(props, leg),
        minutes: leg.minutes,
        done: false,
        splittable: false,
      });
      transfers.set(id, leg);
    }
    tasks.push(task);
    if (task.placeId) lastPlace = task.placeId;
  }
  return { tasks, transfers, error: "" };
}
function transferTitle(props: ItineraryProps, leg: Transfer) {
  return `${leg.mode}: ${props.locations.find((place) => place.id === leg.from)?.name} → ${props.locations.find((place) => place.id === leg.to)?.name}`;
}
export function scheduleTransfers(
  props: ItineraryProps,
  draft: ItineraryDraft,
) {
  const { tasks, transfers, error } = withTransfers(props, draft);
  if (error) return { result: null, transfers, error };
  for (let attempt = 0; attempt < 30; attempt++) {
    const result = calculateSchedule(draft.days, tasks);
    let changed = false;
    for (const allocation of result?.allocations ?? []) {
      const leg = transfers.get(allocation.task.id);
      if (!leg || !allocation.slots.length) continue;
      const date =
        draft.days[Math.floor(allocation.slots[0].start / 1440)].date;
      if (leg.validOn === date) continue;
      const replacement = props.legs.find(
        (item) =>
          item.from === leg.from && item.to === leg.to && item.validOn === date,
      );
      if (!replacement)
        return {
          result: null,
          transfers,
          error:
            "Travel estimates were supplied for a different date. Refresh them in chat before relying on this plan.",
        };
      transfers.set(allocation.task.id, replacement);
      const task = tasks.find((task) => task.id === allocation.task.id)!;
      task.minutes = replacement.minutes;
      task.title = transferTitle(props, replacement);
      changed = true;
    }
    if (!changed) return { result, transfers, error: "" };
  }
  return {
    result: null,
    transfers,
    error:
      "Changing travel estimates keep shifting this plan between dates. Resolve the travel times in chat before exporting.",
  };
}
