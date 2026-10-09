import {
  useStateField,
  type ComponentRenderProps,
} from "@openuidev/react-lang";
import type { z } from "zod/v4";
import type { Checklist } from "@/lib/openui/catalog";
import { cn } from "@/lib/utils";
import { useOpenUIDisabled } from "../interactionContext";

export function ChecklistView({
  props,
  statementId,
}: ComponentRenderProps<z.infer<typeof Checklist.props>>) {
  const field = useStateField<number[]>(
    `checklist:${statementId ?? props.title}`,
    [],
  );
  const completed = new Set<number>(
    Array.isArray(field.value)
      ? field.value.filter(
          (value): value is number => typeof value === "number",
        )
      : [],
  );
  const disabled = useOpenUIDisabled();
  const items = props.items?.slice(0, 12) ?? [];
  function toggle(index: number) {
    if (disabled) return;
    const next = new Set(completed);
    if (next.has(index)) next.delete(index);
    else next.add(index);
    field.setValue([...next]);
  }
  return (
    <section>
      <div className="mb-4 flex items-center justify-between gap-2">
        <h3 className="text-sm font-semibold text-zinc-800">{props.title}</h3>
        <span className="text-xs text-zinc-500" aria-live="polite">
          {completed.size} of {items.length} complete
        </span>
      </div>
      <div className="divide-y divide-zinc-100">
        {items.map((item, index) => (
          <label
            key={index}
            className="flex cursor-pointer gap-3 py-4 first:pt-0 last:pb-0"
          >
            <input
              type="checkbox"
              disabled={disabled}
              checked={completed.has(index)}
              onChange={() => toggle(index)}
              className="mt-0.5 size-4 shrink-0 accent-purple-500"
            />
            <div>
              <span
                className={cn(
                  "text-sm font-medium text-zinc-800",
                  completed.has(index) && "text-zinc-400 line-through",
                )}
              >
                {item.title}
              </span>
              <p className="mt-1 text-xs leading-relaxed text-zinc-500">
                {item.detail}
              </p>
            </div>
          </label>
        ))}
      </div>
    </section>
  );
}
