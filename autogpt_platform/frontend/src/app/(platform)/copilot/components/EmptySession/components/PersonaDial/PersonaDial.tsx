"use client";

import {
  CaretLeftIcon,
  CaretRightIcon,
  MagnifyingGlassIcon,
} from "@phosphor-icons/react";
import { useLayoutEffect, useRef, useState } from "react";
import type { Persona } from "../../personas";
import { DialRing } from "./DialRing";

interface Props {
  personas: Persona[];
  selectedIndex: number;
  onSelect: (index: number) => void;
  onClose: () => void;
}

// Selection mode: a dial whose lowest point sits on the avatar, plus a search
// box pinned near the top of the screen that narrows the ring as you type.
export function PersonaDial({
  personas,
  selectedIndex,
  onSelect,
  onClose,
}: Props) {
  const [query, setQuery] = useState("");
  const rootRef = useRef<HTMLDivElement | null>(null);
  // Distance from the viewport top to the dial anchor (the avatar centre).
  // Needed to pin children N px from the viewport top: `fixed` is unusable
  // here because transformed framer-motion ancestors re-anchor it.
  const [anchorTop, setAnchorTop] = useState<number | null>(null);

  useLayoutEffect(() => {
    const rect = rootRef.current?.getBoundingClientRect();
    if (rect) setAnchorTop(rect.top);
  }, []);

  const needle = query.trim().toLowerCase();
  const filtered = personas.filter((p) =>
    `${p.name} ${p.role}`.toLowerCase().includes(needle),
  );
  const selectedInFiltered = Math.max(
    0,
    filtered.findIndex((p) => p.id === personas[selectedIndex].id),
  );

  function handleRingSelect(filteredIndex: number) {
    onSelect(personas.indexOf(filtered[filteredIndex]));
  }

  function handleArrow(direction: 1 | -1) {
    if (filtered.length === 0) return;
    const next =
      (selectedInFiltered + direction + filtered.length) % filtered.length;
    onSelect(personas.indexOf(filtered[next]));
  }

  return (
    <div
      ref={rootRef}
      role="listbox"
      aria-label="Choose a persona"
      data-persona-picker
      className="absolute left-1/2 top-16 z-0"
    >
      {anchorTop !== null && (
        <div
          className="absolute left-0 z-50 flex -translate-x-1/2 items-center gap-3"
          style={{ top: 16 - anchorTop }}
        >
          <button
            type="button"
            aria-label="Previous persona"
            onClick={() => handleArrow(-1)}
            className="flex size-10 items-center justify-center rounded-full border border-zinc-200 bg-white text-zinc-700 shadow-sm transition-colors hover:bg-zinc-50"
          >
            <CaretLeftIcon size={18} />
          </button>
          <div className="flex items-center gap-2 rounded-full border border-zinc-200 bg-white px-4 py-2 shadow-sm">
            <MagnifyingGlassIcon size={16} className="text-zinc-400" />
            <input
              autoFocus
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && filtered.length > 0) {
                  onSelect(personas.indexOf(filtered[0]));
                  onClose();
                }
              }}
              placeholder="Search personas..."
              aria-label="Search personas"
              className="w-48 bg-transparent text-base outline-none placeholder:text-zinc-400"
            />
          </div>
          <button
            type="button"
            aria-label="Next persona"
            onClick={() => handleArrow(1)}
            className="flex size-10 items-center justify-center rounded-full border border-zinc-200 bg-white text-zinc-700 shadow-sm transition-colors hover:bg-zinc-50"
          >
            <CaretRightIcon size={18} />
          </button>
        </div>
      )}

      {filtered.length > 0 ? (
        // Keyed by query so the ring's rotation resets (and items re-stagger)
        // whenever the visible set changes.
        <DialRing
          key={needle}
          personas={filtered}
          selectedIndex={selectedInFiltered}
          onSelect={handleRingSelect}
          onClose={onClose}
        />
      ) : (
        anchorTop !== null && (
          <p
            className="absolute left-0 z-50 w-max -translate-x-1/2 text-sm text-zinc-500"
            style={{ top: 76 - anchorTop }}
          >
            No personas match &quot;{query}&quot;
          </p>
        )
      )}
    </div>
  );
}
