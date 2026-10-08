import * as L from "leaflet";
import type { placeCoordinates } from "./helpers";

type PlacePoint = ReturnType<typeof placeCoordinates>[number];

export function createPlaceMarker(
  point: PlacePoint,
  onSelect: (index: number) => void,
) {
  const badge = document.createElement("span");
  badge.className =
    "flex size-8 items-center justify-center rounded-full border-2 border-white bg-purple-600 text-sm font-semibold text-white shadow-md ring-purple-300";
  badge.textContent = String(point.index + 1);
  const tooltip = document.createElement("span");
  tooltip.textContent = point.place.name;
  const marker = L.marker(point.coordinates, {
    title: point.place.name,
    alt: point.place.name,
    icon: L.divIcon({
      html: badge,
      className: "",
      iconSize: [32, 32],
      iconAnchor: [16, 16],
    }),
    keyboard: true,
  });
  marker.on("add", () =>
    marker.getElement()?.setAttribute("aria-label", point.place.name),
  );
  return marker.bindTooltip(tooltip).on("click", () => onSelect(point.index));
}

export function fitPlaces(map: L.Map, points: PlacePoint[]) {
  if (!points.length) return;
  map.fitBounds(L.latLngBounds(points.map((point) => point.coordinates)), {
    padding: [28, 28],
    maxZoom: 13,
    animate: false,
  });
}

export function createMap(
  container: HTMLDivElement,
  points: PlacePoint[],
  onSelect: (index: number) => void,
  onTileError: () => void,
) {
  const map = L.map(container, {
    scrollWheelZoom: false,
    maxZoom: 18,
    zoomAnimation: false,
    fadeAnimation: false,
    markerZoomAnimation: false,
  });
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18,
    referrerPolicy: "strict-origin-when-cross-origin",
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors',
  })
    .on("tileerror", onTileError)
    .addTo(map);
  const markers = points.map((point) =>
    createPlaceMarker(point, onSelect).addTo(map),
  );
  fitPlaces(map, points);
  const observer = new ResizeObserver(() =>
    map.invalidateSize({ animate: false }),
  );
  observer.observe(container);
  return {
    map,
    points,
    markers,
    destroy() {
      observer.disconnect();
      map.remove();
    },
  };
}

export function selectPlace(
  instance: ReturnType<typeof createMap>,
  selectedIndex: number,
) {
  const selected = instance.points.find(
    (point) => point.index === selectedIndex,
  );
  if (selected)
    instance.map.setView(
      selected.coordinates,
      Math.max(13, instance.map.getZoom()),
      { animate: false },
    );
  else fitPlaces(instance.map, instance.points);
  instance.markers.forEach((marker, index) => {
    const active = instance.points[index].index === selectedIndex;
    marker.getElement()?.setAttribute("aria-pressed", String(active));
    marker.setZIndexOffset(active ? 1000 : 0);
    marker.getElement()?.firstElementChild?.classList.toggle("ring-4", active);
  });
}
