"use client";

import "leaflet/dist/leaflet.css";
import type { IndexedLocation } from "./helpers";
import { useMapCanvas } from "./useMapCanvas";

interface Props {
  locations: IndexedLocation[];
  selectedIndex: number;
  onSelect: (index: number) => void;
  title: string;
}

export function MapCanvas({
  locations,
  selectedIndex,
  onSelect,
  title,
}: Props) {
  const { containerRef, error } = useMapCanvas(
    locations,
    selectedIndex,
    onSelect,
  );
  return (
    <div className="relative isolate">
      <div
        ref={containerRef}
        className="h-72 w-full bg-zinc-100 sm:h-80"
        aria-label={`${title} map`}
      />
      {error && (
        <p
          role="status"
          className="border-y border-amber-200 bg-amber-50 p-3 text-xs text-amber-800"
        >
          Map tiles could not load. You can still select and discuss places from
          the list below.
        </p>
      )}
    </div>
  );
}
