import { useEffect, useRef, useState } from "react";
import { loadGoogleMaps, onMapsAuthFailure } from "./googleMapsLoader";
import type { MapPin } from "./helpers";

function pinPopup(pin: MapPin) {
  const wrapper = document.createElement("div");
  const name = document.createElement("strong");
  name.textContent = pin.name;
  wrapper.appendChild(name);
  if (pin.address) {
    const address = document.createElement("div");
    address.textContent = pin.address;
    wrapper.appendChild(address);
  }
  return wrapper;
}

export function useMapEmbed(pins: MapPin[]) {
  const apiKey = process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY ?? "";
  const containerRef = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "failed">(
    apiKey ? "loading" : "failed",
  );
  const pinsKey = JSON.stringify(pins);

  useEffect(() => {
    const container = containerRef.current;
    if (!apiKey || !container) return;
    let cancelled = false;
    let rejected = false;
    const unsubscribe = onMapsAuthFailure(() => {
      rejected = true;
      if (!cancelled) setStatus("failed");
    });
    loadGoogleMaps(apiKey)
      .then((google) => {
        if (cancelled || rejected) return;
        const { Map, LatLngBounds, InfoWindow, marker } = google.maps;
        const map = new Map(container, {
          mapId: "DEMO_MAP_ID",
          center: pins[0],
          zoom: 14,
          mapTypeControl: false,
          streetViewControl: false,
        });
        const info = new InfoWindow();
        const bounds = new LatLngBounds();
        pins.forEach((pin) => {
          const pinMarker = new marker.AdvancedMarkerElement({
            map,
            position: { lat: pin.lat, lng: pin.lng },
            title: pin.name,
          });
          pinMarker.addListener("click", () => {
            info.setContent(pinPopup(pin));
            info.open({ map, anchor: pinMarker });
          });
          bounds.extend({ lat: pin.lat, lng: pin.lng });
        });
        if (pins.length > 1) map.fitBounds(bounds, 48);
        setStatus("ready");
      })
      .catch(() => {
        if (!cancelled) setStatus("failed");
      });
    return () => {
      cancelled = true;
      unsubscribe();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiKey, pinsKey]);

  return { containerRef, status };
}
