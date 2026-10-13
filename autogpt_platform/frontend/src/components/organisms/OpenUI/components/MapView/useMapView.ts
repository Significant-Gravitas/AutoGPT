import { useState } from "react";
import {
  useSetFieldValue,
  useStateField,
  useTriggerAction,
} from "@openuidev/react-lang";
import type { MapLocation } from "@/lib/openui/catalog-sections";
import { validLocations } from "./helpers";

export function useMapView(locations: MapLocation[], statementID: string) {
  const [filter, setFilter] = useState("");
  const field = useStateField<number>(`map-selection:${statementID}`, -1);
  const triggerAction = useTriggerAction();
  const setFieldValue = useSetFieldValue();
  const places = validLocations(locations);
  const query = filter.trim().toLocaleLowerCase();
  const visiblePlaces = places.filter(({ place }) =>
    `${place.name} ${place.category} ${place.detail}`
      .toLocaleLowerCase()
      .includes(query),
  );
  const selected = visiblePlaces.find(({ index }) => index === field.value);

  function discussPlace() {
    if (!selected) return;
    const formName = `map:${statementID}`;
    for (const [name, value] of Object.entries(selected.place)) {
      setFieldValue(formName, "Map", name, value, false);
    }
    triggerAction(
      "Help me explore or compare this selected place using the context of our conversation.",
      formName,
    );
  }

  function showAll() {
    setFilter("");
    field.setValue(-1);
  }

  return {
    filter,
    setFilter,
    places,
    visiblePlaces,
    selected,
    select: field.setValue,
    showAll,
    discussPlace,
  };
}
