"use client";

import dynamic from "next/dynamic";
import {
  useIsStreaming,
  type ComponentRenderProps,
} from "@openuidev/react-lang";
import type { z } from "zod/v4";
import type { Map } from "@/lib/openui/catalog-sections";
import { useMapView } from "./useMapView";
import { MapControls } from "./MapControls";
import { PlaceList } from "./PlaceList";
import { SelectedPlace } from "./SelectedPlace";

const MapCanvas = dynamic(
  () => import("./MapCanvas").then((module) => module.MapCanvas),
  {
    ssr: false,
    loading: () => (
      <div
        className="flex h-72 items-center justify-center bg-zinc-100 text-sm text-zinc-500"
        role="status"
      >
        Loading map…
      </div>
    ),
  },
);

export function MapView({
  props,
  statementId,
}: ComponentRenderProps<z.infer<typeof Map.props>>) {
  const streaming = useIsStreaming();
  const ui = useMapView(props.locations ?? [], statementId ?? props.title);
  return (
    <section
      aria-label={props.title}
      className="min-w-0 overflow-hidden rounded-xl border border-zinc-200 bg-white"
    >
      <MapControls
        title={props.title}
        description={props.description}
        filter={ui.filter}
        onFilter={ui.setFilter}
        onShowAll={ui.showAll}
      />
      {!streaming && ui.visiblePlaces.length > 0 && (
        <MapCanvas
          locations={ui.visiblePlaces}
          selectedIndex={ui.selected?.index ?? -1}
          onSelect={ui.select}
          title={props.title}
        />
      )}
      <div className="space-y-3 p-4">
        <p className="text-xs text-zinc-500" role="status">
          {streaming
            ? "Preparing locations…"
            : `${ui.visiblePlaces.length} of ${ui.places.length} places`}
        </p>
        <PlaceList
          places={ui.visiblePlaces}
          selectedIndex={ui.selected?.index ?? -1}
          onSelect={ui.select}
        />
        {ui.selected && (
          <SelectedPlace
            place={ui.selected.place}
            onDiscuss={ui.discussPlace}
          />
        )}
      </div>
    </section>
  );
}
