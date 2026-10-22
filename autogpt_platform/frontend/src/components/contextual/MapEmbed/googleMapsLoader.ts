export interface LatLngLiteral {
  lat: number;
  lng: number;
}

interface MapInstance {
  fitBounds(bounds: unknown, padding?: number): void;
  setCenter(center: LatLngLiteral): void;
  setZoom(zoom: number): void;
}

interface MarkerInstance {
  addListener(event: string, handler: () => void): void;
}

interface InfoWindowInstance {
  setContent(content: Node): void;
  open(options: { map: MapInstance; anchor: MarkerInstance }): void;
}

export interface GoogleMapsApi {
  maps: {
    Map: new (el: HTMLElement, options: Record<string, unknown>) => MapInstance;
    LatLngBounds: new () => { extend(point: LatLngLiteral): void };
    InfoWindow: new () => InfoWindowInstance;
    marker: {
      AdvancedMarkerElement: new (options: {
        map: MapInstance;
        position: LatLngLiteral;
        title?: string;
      }) => MarkerInstance;
    };
    importLibrary(name: string): Promise<unknown>;
  };
}

declare global {
  interface Window {
    google?: Partial<GoogleMapsApi>;
  }
}

let loading: Promise<GoogleMapsApi> | null = null;

export function loadGoogleMaps(apiKey: string): Promise<GoogleMapsApi> {
  if (loading) return loading;
  loading = new Promise<GoogleMapsApi>((resolve, reject) => {
    const script = document.createElement("script");
    script.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(apiKey)}&loading=async&v=weekly`;
    script.async = true;
    script.onerror = () => reject(new Error("Google Maps failed to load"));
    script.onload = async () => {
      try {
        const api = window.google as GoogleMapsApi | undefined;
        if (!api?.maps) throw new Error("Google Maps failed to load");
        await api.maps.importLibrary("maps");
        await api.maps.importLibrary("marker");
        resolve(api);
      } catch (error) {
        reject(error);
      }
    };
    document.head.appendChild(script);
  }).catch((error: unknown) => {
    loading = null;
    throw error;
  });
  return loading;
}
