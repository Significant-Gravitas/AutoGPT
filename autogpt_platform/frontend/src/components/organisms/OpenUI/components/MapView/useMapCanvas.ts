import { useEffect, useRef, useState } from "react";
import {
  indexedLocationsSchema,
  placeCoordinates,
  type IndexedLocation,
} from "./helpers";
import { createMap, selectPlace } from "./leafletHelpers";

export function useMapCanvas(
  locations: IndexedLocation[],
  selectedIndex: number,
  onSelect: (index: number) => void,
) {
  const containerRef = useRef<HTMLDivElement>(null);
  const instanceRef = useRef<ReturnType<typeof createMap> | null>(null);
  const selectRef = useRef(onSelect);
  const [error, setError] = useState(false);
  const locationKey = JSON.stringify(locations);

  useEffect(() => {
    selectRef.current = onSelect;
  }, [onSelect]);

  useEffect(() => {
    if (!containerRef.current) return;
    const points = placeCoordinates(
      indexedLocationsSchema.parse(JSON.parse(locationKey)),
    );
    setError(false);
    const instance = createMap(
      containerRef.current,
      points,
      (index) => selectRef.current(index),
      () => setError(true),
    );
    instanceRef.current = instance;
    return () => {
      instance.destroy();
      instanceRef.current = null;
    };
  }, [locationKey]);

  useEffect(() => {
    const instance = instanceRef.current;
    if (!instance) return;
    selectPlace(instance, selectedIndex);
  }, [selectedIndex, locationKey]);

  return { containerRef, error };
}
