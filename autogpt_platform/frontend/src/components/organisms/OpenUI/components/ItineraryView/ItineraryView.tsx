import dynamic from "next/dynamic";
import type { ComponentRenderProps } from "@openuidev/react-lang";
import type { ItineraryProps } from "@/lib/openui/itinerary";
import { scheduleTime } from "@/lib/openui/schedule";
import { Button } from "@/components/atoms/Button/Button";
import { useOpenUIDisabled } from "../../interactionContext";
import { ScheduleTaskRow } from "../ScheduleView/ScheduleTaskRow";
import { ControlValidity } from "../ControlValidity";
import { AvailabilityEditor } from "./AvailabilityEditor";
import { useItinerary } from "./useItinerary";

const MapCanvas = dynamic(
  () => import("../MapView/MapCanvas").then((module) => module.MapCanvas),
  { ssr: false, loading: () => <p className="text-sm">Loading map…</p> },
);

export function ItineraryView({ props }: ComponentRenderProps<ItineraryProps>) {
  const view = useItinerary(props);
  const disabled = useOpenUIDisabled();
  const { draft, plan } = view;
  const selectedIndex = props.locations.findIndex(
    (place) => place.id === draft.selected,
  );
  return (
    <fieldset disabled={disabled} className="min-w-0 space-y-4">
      <legend className="font-semibold">{props.title}</legend>
      <p className="text-xs text-muted-foreground">
        Proposed itinerary · {draft.timeZone} · Downloads do not make bookings
      </p>
      <MapCanvas
        title={props.title}
        locations={props.locations.map((place, index) => ({ index, place }))}
        selectedIndex={selectedIndex}
        onSelect={(index) => {
          if (!disabled) view.change({ selected: props.locations[index].id });
        }}
      />
      <div className="flex flex-wrap gap-2" aria-label="Itinerary places">
        {props.locations.map((place, index) => (
          <Button
            key={place.id}
            type="button"
            size="small"
            variant={place.id === draft.selected ? "primary" : "secondary"}
            aria-pressed={place.id === draft.selected}
            onClick={() => view.change({ selected: place.id })}
          >
            {index + 1}. {place.name}
          </Button>
        ))}
      </div>
      {selectedIndex >= 0 && (
        <p className="text-sm" aria-live="polite">
          {props.locations[selectedIndex].detail}
        </p>
      )}
      <AvailabilityEditor draft={draft} onChange={view.change} />
      <ControlValidity label={props.title} error={plan.error} />
      {plan.result && (
        <p role="status" className="text-sm">
          {plan.result.required} min remaining · {plan.result.capacity} min
          available ·{" "}
          {plan.result.feasible
            ? `${plan.result.buffer} min spare`
            : `${plan.result.unscheduled} min unscheduled. This plan does not fit.`}
        </p>
      )}
      <ol className="space-y-3">
        {draft.tasks.map((task, index) => {
          const allocation = plan.result?.allocations.find(
            (item) => item.task.id === task.id,
          );
          return (
            <ScheduleTaskRow
              key={task.id}
              highlighted={task.placeId === draft.selected}
              task={task}
              disabled={disabled}
              first={index === 0}
              last={index === draft.tasks.length - 1}
              times={
                allocation?.slots.map(
                  (slot) =>
                    `${scheduleTime(slot.start, draft.days)}–${scheduleTime(slot.end, draft.days)}`,
                ) ?? []
              }
              conflict={allocation?.conflict ?? ""}
              onMove={(direction) => view.move(index, direction)}
              onChange={view.changeTask}
            >
              {task.placeId && (
                <Button
                  type="button"
                  size="small"
                  variant="ghost"
                  onClick={() => view.change({ selected: task.placeId! })}
                  aria-label={`Show ${task.title} on map`}
                >
                  Show on map
                </Button>
              )}
            </ScheduleTaskRow>
          );
        })}
      </ol>
      {plan.result && (
        <ul
          aria-label="Transfers"
          className="space-y-2 text-xs text-muted-foreground"
        >
          {plan.result.allocations
            .filter(
              (item) => !draft.tasks.some((task) => task.id === item.task.id),
            )
            .map((item) => (
              <li key={item.task.id}>
                {item.task.title} · {item.task.minutes} min
                {item.conflict ? ` · ${item.conflict}` : ""}
                {plan.transfers?.get(item.task.id)?.source && (
                  <a
                    href={plan.transfers.get(item.task.id)!.source!.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="ml-2 text-primary underline"
                  >
                    {plan.transfers.get(item.task.id)!.source!.label}
                  </a>
                )}
              </li>
            ))}
        </ul>
      )}
      <Button
        type="button"
        size="small"
        variant="secondary"
        disabled={!plan.result?.feasible || !!plan.error}
        onClick={view.download}
      >
        Download current calendar
      </Button>
      {view.exportError && (
        <p role="alert" className="text-sm text-destructive">
          {view.exportError}
        </p>
      )}
    </fieldset>
  );
}
