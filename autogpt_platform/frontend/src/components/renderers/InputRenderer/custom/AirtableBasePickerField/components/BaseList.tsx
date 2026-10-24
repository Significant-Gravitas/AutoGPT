"use client";

import type { AirtableBase } from "@/app/api/__generated__/models/airtableBase";
import { Icon } from "@/components/atoms/Icon/Icon";
import { isKey } from "@/lib/keyboard";
import { cn } from "@/lib/utils";
import { Tick02Icon } from "@hugeicons/core-free-icons";
import { useRef } from "react";

interface Props {
  bases: AirtableBase[];
  value: string;
  labelID: string;
  onChange: (id: string) => void;
}

/** The bases as one radiogroup: a single tab stop, arrows move and select.
 *  A view-only base is listed so the user recognises it, but can't be picked. */
export function BaseList({ bases, value, labelID, onChange }: Props) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const pickable = bases.filter((base) => base.can_edit);
  const selected = pickable.findIndex((base) => base.id === value);
  const activeID = pickable[selected === -1 ? 0 : selected]?.id;

  function handleKeyDown(event: React.KeyboardEvent, id: string) {
    const step = isKey(event, "ArrowDown", "ArrowRight")
      ? 1
      : isKey(event, "ArrowUp", "ArrowLeft")
        ? -1
        : 0;
    if (step === 0) return;
    event.preventDefault();
    const index = pickable.findIndex((base) => base.id === id);
    const next = pickable[(index + step + pickable.length) % pickable.length];
    onChange(next.id);
    refs.current[bases.indexOf(next)]?.focus();
  }

  return (
    <div
      role="radiogroup"
      aria-labelledby={labelID}
      aria-describedby={`${labelID}-hint`}
      aria-required="true"
      className="flex max-h-72 flex-col gap-1.5 overflow-y-auto p-0.5"
    >
      {bases.map((base, index) => {
        const isSelected = base.id === value;
        return (
          <button
            key={base.id}
            ref={(element) => {
              refs.current[index] = element;
            }}
            type="button"
            role="radio"
            aria-checked={isSelected}
            aria-disabled={!base.can_edit}
            disabled={!base.can_edit}
            tabIndex={base.id === activeID ? 0 : -1}
            onClick={() => onChange(base.id)}
            onKeyDown={(event) => handleKeyDown(event, base.id)}
            className={cn(
              "flex w-full min-w-0 items-center gap-3 rounded-2xl px-4 py-2.5 text-start text-sm leading-snug transition-colors",
              isSelected
                ? "bg-white text-zinc-900 ring-2 ring-zinc-800"
                : "bg-zinc-50 text-zinc-700 ring-1 ring-zinc-100 hover:bg-zinc-100",
              !base.can_edit &&
                "cursor-not-allowed text-zinc-400 hover:bg-zinc-50",
            )}
          >
            <span className="min-w-0 flex-1 break-words">{base.name}</span>
            {!base.can_edit && (
              <span className="shrink-0 text-xs text-zinc-500">View only</span>
            )}
            {isSelected && (
              <Icon icon={Tick02Icon} size={16} className="shrink-0" />
            )}
          </button>
        );
      })}
    </div>
  );
}
