export type DeviceMarker = {
    id: string;
    name: string;
    x: number;
    y: number;
    status?: 'normal' | 'warning' | 'alarm';
    meta?: Record<string, unknown>;
};
export type TiledImageInfo = {
    id: string;
    name: string;
    width: number;
    height: number;
    tileSize: number;
    tileOverlap: number;
    levels: number;
    dziUrl?: string;
};
export declare const useImageStore: import("pinia").StoreDefinition<"image", {
    image: TiledImageInfo | null;
    markers: DeviceMarker[];
    selectedMarkerId: string | null;
    pendingMarker: null | {
        id: string;
        name: string;
    };
}, {
    selectedMarker(state: {
        image: {
            id: string;
            name: string;
            width: number;
            height: number;
            tileSize: number;
            tileOverlap: number;
            levels: number;
            dziUrl?: string;
        };
        markers: {
            id: string;
            name: string;
            x: number;
            y: number;
            status?: "normal" | "warning" | "alarm";
            meta?: Record<string, unknown>;
        }[];
        selectedMarkerId: string | null;
        pendingMarker: {
            id: string;
            name: string;
        };
    } & import("pinia").PiniaCustomStateProperties<{
        image: TiledImageInfo | null;
        markers: DeviceMarker[];
        selectedMarkerId: string | null;
        pendingMarker: null | {
            id: string;
            name: string;
        };
    }>): {
        id: string;
        name: string;
        x: number;
        y: number;
        status?: "normal" | "warning" | "alarm";
        meta?: Record<string, unknown>;
    };
}, {
    setImage(info: TiledImageInfo | null): void;
    addMarker(marker: DeviceMarker): void;
    startPlacingMarker(id: string, name: string): void;
    cancelPlacingMarker(): void;
    completePlacingMarker(xNormalized: number, yNormalized: number): void;
    updateMarker(id: string, patch: Partial<DeviceMarker>): void;
    removeMarker(id: string): void;
    selectMarker(id: string | null): void;
}>;
