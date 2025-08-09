export type AlertEvent = {
    id: string;
    deviceId: string;
    type: string;
    level: '低' | '中' | '高';
    timestamp: string;
    description?: string;
    resolved?: boolean;
};
export declare const useAlertStore: import("pinia").StoreDefinition<"alert", {
    events: AlertEvent[];
    blinkingDeviceIds: Set<string>;
}, {
    activeEvents(state: {
        events: {
            id: string;
            deviceId: string;
            type: string;
            level: "\u4F4E" | "\u4E2D" | "\u9AD8";
            timestamp: string;
            description?: string;
            resolved?: boolean;
        }[];
        blinkingDeviceIds: Set<string> & Omit<Set<string>, keyof Set<any>>;
    } & import("pinia").PiniaCustomStateProperties<{
        events: AlertEvent[];
        blinkingDeviceIds: Set<string>;
    }>): {
        id: string;
        deviceId: string;
        type: string;
        level: "\u4F4E" | "\u4E2D" | "\u9AD8";
        timestamp: string;
        description?: string;
        resolved?: boolean;
    }[];
}, {
    triggerAlert(event: AlertEvent): void;
    resolveAlert(eventId: string): void;
    clearBlink(deviceId: string): void;
}>;
