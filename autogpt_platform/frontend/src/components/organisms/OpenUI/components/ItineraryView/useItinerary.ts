import { useContext, useState } from "react";
import { OpenUIInstanceContext } from "../../interactionContext";
import {
  itineraryDraft,
  planItinerary,
  type ItineraryProps,
  type ItineraryDraft,
} from "@/lib/openui/itinerary";
import { asNumber } from "@/lib/openui/calculations";
import type { ScheduleDraft } from "@/lib/openui/schedule";
import { itineraryCalendar } from "@/lib/openui/calendar-export";
import { downloadText } from "@/lib/openui/download";
import { useFieldView } from "../useFieldView";
import { useDerivedField } from "../useDerivedField";

export function useItinerary(props: ItineraryProps) {
  const instance = useContext(OpenUIInstanceContext);
  const initial = {
    days: props.days,
    tasks: props.tasks,
    timeZone: props.timeZone,
    selected: props.locations[0].id,
  };
  const field = useFieldView<ItineraryDraft>(`${props.name}__draft`, initial);
  const parsed = itineraryDraft.safeParse(field.value);
  const draft = parsed.success ? parsed.data : initial;
  const plan = planItinerary(props, draft);
  const [exportError, setExportError] = useState("");
  useDerivedField(props.name, {
    ...draft,
    tasks: draft.tasks.map((task) => ({
      ...task,
      minutes: task.done ? null : asNumber(task.minutes),
    })),
    feasible: plan.result?.feasible ?? false,
    issue: plan.error,
  });
  useDerivedField(`${props.name}__valid`, !plan.error);
  useDerivedField(
    `${props.name}_required_minutes`,
    plan.result?.required ?? null,
  );
  useDerivedField(
    `${props.name}_capacity_minutes`,
    plan.result?.capacity ?? null,
  );
  useDerivedField(`${props.name}_buffer_minutes`, plan.result?.buffer ?? null);
  useDerivedField(
    `${props.name}_unscheduled_minutes`,
    plan.result?.unscheduled ?? null,
  );
  useDerivedField(`${props.name}_feasible`, plan.result?.feasible ?? false);
  function change(patch: Partial<ItineraryDraft>) {
    field.setValue({ ...draft, ...patch });
    setExportError("");
  }
  function move(index: number, direction: number) {
    const tasks = [...draft.tasks];
    [tasks[index], tasks[index + direction]] = [
      tasks[index + direction],
      tasks[index],
    ];
    change({ tasks });
  }
  function changeTask(task: ScheduleDraft) {
    change({
      tasks: draft.tasks.map((item) =>
        item.id === task.id ? { ...item, ...task } : item,
      ),
    });
  }
  function download() {
    try {
      downloadText(
        `${props.name}.ics`,
        itineraryCalendar(props, draft, new Date(), instance),
        "text/calendar;charset=utf-8",
      );
    } catch (error) {
      setExportError(
        error instanceof Error ? error.message : "Calendar export failed.",
      );
    }
  }
  return { draft, plan, change, changeTask, move, download, exportError };
}
