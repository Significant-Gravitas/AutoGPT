import Map from "ol/Map";
import View from "ol/View";
import Overlay from "ol/Overlay";
import TileLayer from "ol/layer/Tile";
import OSM from "ol/source/OSM";
import { fromLonLat } from "ol/proj";
import { boundingExtent } from "ol/extent";
import { defaults as defaultInteractions } from "ol/interaction/defaults";
import { defaults as defaultControls } from "ol/control/defaults";
import type { placeCoordinates } from "./helpers";

type PlacePoint = ReturnType<typeof placeCoordinates>[number];

function projectedPoint(point: PlacePoint) {
  return fromLonLat([point.coordinates[1], point.coordinates[0]]);
}

export function createPlaceMarker(
  point: PlacePoint,
  onSelect: (index: number) => void,
) {
  const button = document.createElement("button");
  button.type = "button";
  button.className =
    "flex size-8 items-center justify-center rounded-full border-2 border-white bg-purple-600 text-sm font-semibold text-white shadow-md ring-purple-300 focus-visible:ring-4";
  button.textContent = String(point.index + 1);
  button.title = point.place.name;
  button.setAttribute("aria-label", point.place.name);
  button.addEventListener("click", () => onSelect(point.index));
  return new Overlay({
    element: button,
    position: projectedPoint(point),
    positioning: "center-center",
    stopEvent: true,
  });
}

export function fitPlaces(map: Map, points: PlacePoint[]) {
  if (!points.length) return;
  map.getView().fit(boundingExtent(points.map(projectedPoint)), {
    padding: [28, 28, 28, 28],
    maxZoom: 13,
    duration: 0,
  });
}

export function createMap(
  container: HTMLDivElement,
  points: PlacePoint[],
  onSelect: (index: number) => void,
  onTileError: () => void,
) {
  const source = new OSM({ maxZoom: 18 });
  source.on("tileloaderror", onTileError);
  const markers = points.map((point) => createPlaceMarker(point, onSelect));
  const map = new Map({
    target: container,
    layers: [new TileLayer({ source })],
    overlays: markers,
    view: new View({
      center: points.length ? projectedPoint(points[0]) : [0, 0],
      zoom: 13,
      maxZoom: 18,
      multiWorld: true,
    }),
    controls: defaultControls({
      rotate: false,
      attributionOptions: { collapsible: false },
    }),
    interactions: defaultInteractions({
      mouseWheelZoom: false,
      altShiftDragRotate: false,
      pinchRotate: false,
    }),
  });
  map.setSize([container.clientWidth || 640, container.clientHeight || 288]);
  fitPlaces(map, points);
  const observer = new ResizeObserver(() => {
    if (container.clientWidth && container.clientHeight) map.updateSize();
  });
  observer.observe(container);
  return {
    map,
    points,
    markers,
    destroy() {
      observer.disconnect();
      source.un("tileloaderror", onTileError);
      map.setTarget(undefined);
      map.dispose();
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
  if (selected) {
    instance.map.getView().setCenter(projectedPoint(selected));
    instance.map
      .getView()
      .setZoom(Math.max(13, instance.map.getView().getZoom() ?? 13));
  } else fitPlaces(instance.map, instance.points);
  instance.markers.forEach((marker, index) => {
    const active = instance.points[index].index === selectedIndex;
    const element = marker.getElement();
    element?.setAttribute("aria-pressed", String(active));
    element?.classList.toggle("ring-4", active);
    element?.parentElement?.classList.toggle("z-10", active);
  });
}
