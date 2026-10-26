import { useEffect, useRef, useState } from "react";
import { loadGoogleMaps, onMapsAuthFailure } from "./googleMapsLoader";
import type { MapPin } from "./helpers";

const LOAD_TIMEOUT_MS = 10000;

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
    const pins = JSON.parse(pinsKey) as MapPin[];
    let cancelled = false;
    let rejected = false;
    const markers: Array<{ map: unknown }> = [];
    let info: { close(): void } | null = null;
    const teardown = () => {
      info?.close();
      markers.forEach((pinMarker) => {
        pinMarker.map = null;
      });
      container.replaceChildren();
    };
    const fail = () => {
      rejected = true;
      teardown();
      if (!cancelled) setStatus("failed");
    };
    const timeout = setTimeout(fail, LOAD_TIMEOUT_MS);
    const unsubscribe = onMapsAuthFailure(fail);
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
        const infoWindow = new InfoWindow();
        info = infoWindow;
        const bounds = new LatLngBounds();
        pins.forEach((pin) => {
          const pinMarker = new marker.AdvancedMarkerElement({
            map,
            position: { lat: pin.lat, lng: pin.lng },
            title: pin.name,
            gmpClickable: true,
          });
          pinMarker.addEventListener("gmp-click", () => {
            infoWindow.setContent(pinPopup(pin));
            infoWindow.open({ map, anchor: pinMarker });
          });
          markers.push(pinMarker);
          bounds.extend({ lat: pin.lat, lng: pin.lng });
        });
        if (pins.length > 1) map.fitBounds(bounds, 48);
        clearTimeout(timeout);
        setStatus("ready");
      })
      .catch(() => {
        if (!cancelled) setStatus("failed");
      });
    return () => {
      cancelled = true;
      clearTimeout(timeout);
      unsubscribe();
      teardown();
    };
  }, [apiKey, pinsKey]);

  return { containerRef, status };
}
