import type { ComponentRenderProps } from "@openuidev/react-lang";
import type { z } from "zod/v4";
import type { Checklist } from "@/lib/openui/catalog";
import { useState } from "react";
import { cn } from "@/lib/utils";

export function ChecklistView({
  props,
}: ComponentRenderProps<z.infer<typeof Checklist.props>>) {
  const [completed, setCompleted] = useState<Set<number>>(new Set());
  const items = props.items?.slice(0, 12) ?? [];
  function toggle(index: number) {
    setCompleted((current) => {
      const next = new Set(current);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  }
  return (
    <section className="rounded-xl border border-zinc-200 bg-white p-5">
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
