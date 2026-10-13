import { useId } from "react";
import { Input } from "@/components/atoms/Input/Input";
import { Button } from "@/components/atoms/Button/Button";
import type { ItineraryDraft } from "@/lib/openui/itinerary";

interface Props {
  draft: ItineraryDraft;
  onChange: (patch: Partial<ItineraryDraft>) => void;
}
export function AvailabilityEditor({ draft, onChange }: Props) {
  const id = useId();
  function dayChange(
    index: number,
    patch: Partial<ItineraryDraft["days"][number]>,
  ) {
    onChange({
      days: draft.days.map((day, i) =>
        i === index ? { ...day, ...patch } : day,
      ),
    });
  }
  return (
    <details>
      <summary className="min-h-11 cursor-pointer py-3 text-sm font-medium">
        Edit dates and available times
      </summary>
      <div className="space-y-3">
        <Input
          id={`${id}-zone`}
          label="Time zone"
          value={draft.timeZone}
          maxLength={100}
          onChange={(event) => onChange({ timeZone: event.target.value })}
        />
        {draft.days.map((day, index) => (
          <fieldset key={index} className="space-y-2">
            <legend className="text-sm font-medium">{day.label}</legend>
            <Input
              label=""
              id={`${id}-${index}`}
              aria-label={`${day.label} date`}
              type="date"
              value={day.date}
              onChange={(event) =>
                dayChange(index, { date: event.target.value })
              }
            />
            {day.windows.map((window, windowIndex) => (
              <div key={windowIndex} className="flex flex-wrap items-end gap-2">
                <input
                  className="min-h-11 rounded-md border bg-background p-2 text-sm"
                  id={`${id}-${index}-${windowIndex}-start`}
                  aria-label={`${day.label} window ${windowIndex + 1} start`}
                  type="time"
                  value={window.start}
                  onChange={(event) =>
                    dayChange(index, {
                      windows: day.windows.map((item, i) =>
                        i === windowIndex
                          ? { ...item, start: event.target.value }
                          : item,
                      ),
                    })
                  }
                />
                <input
                  className="min-h-11 rounded-md border bg-background p-2 text-sm"
                  id={`${id}-${index}-${windowIndex}-end`}
                  aria-label={`${day.label} window ${windowIndex + 1} end`}
                  type="time"
                  value={window.end}
                  onChange={(event) =>
                    dayChange(index, {
                      windows: day.windows.map((item, i) =>
                        i === windowIndex
                          ? { ...item, end: event.target.value }
                          : item,
                      ),
                    })
                  }
                />
                <Button
                  type="button"
                  size="small"
                  variant="secondary"
                  disabled={day.windows.length === 1}
                  aria-label={`Remove ${day.label} window ${windowIndex + 1}`}
                  onClick={() =>
                    dayChange(index, {
                      windows: day.windows.filter((_, i) => i !== windowIndex),
                    })
                  }
                >
                  Remove
                </Button>
              </div>
            ))}
            <Button
              type="button"
              size="small"
              variant="secondary"
              disabled={day.windows.length >= 4}
              onClick={() =>
                dayChange(index, {
                  windows: [...day.windows, { start: "", end: "" }],
                })
              }
            >
              Add time window
            </Button>
          </fieldset>
        ))}
      </div>
    </details>
  );
}
