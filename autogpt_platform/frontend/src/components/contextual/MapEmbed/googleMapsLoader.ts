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
  addEventListener(event: string, handler: () => void): void;
  map: MapInstance | null;
}

interface InfoWindowInstance {
  setContent(content: Node): void;
  open(options: { map: MapInstance; anchor: MarkerInstance }): void;
  close(): void;
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
        gmpClickable?: boolean;
      }) => MarkerInstance;
    };
    importLibrary(name: string): Promise<unknown>;
  };
}

interface MapsWindow {
  google?: Partial<GoogleMapsApi>;
  gm_authFailure?: () => void;
}

let loading: Promise<GoogleMapsApi> | null = null;
let authFailed = false;
const authFailureListeners = new Set<() => void>();

function handleAuthFailure() {
  authFailed = true;
  authFailureListeners.forEach((listener) => listener());
}

// Maps JS reports a rejected key (referrer, billing, API access) through this
// global callback after the script has loaded, not through the load promise.
export function onMapsAuthFailure(listener: () => void) {
  if (authFailed) listener();
  authFailureListeners.add(listener);
  return function unsubscribe() {
    authFailureListeners.delete(listener);
  };
}

export function loadGoogleMaps(apiKey: string): Promise<GoogleMapsApi> {
  if (loading) return loading;
  loading = new Promise<GoogleMapsApi>((resolve, reject) => {
    (window as MapsWindow).gm_authFailure = handleAuthFailure;
    const script = document.createElement("script");
    script.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(apiKey)}&loading=async&v=weekly`;
    script.async = true;
    script.onerror = () => reject(new Error("Google Maps failed to load"));
    script.onload = async () => {
      try {
        const api = (window as MapsWindow).google as GoogleMapsApi | undefined;
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
