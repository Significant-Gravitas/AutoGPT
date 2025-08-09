import { defineStore } from 'pinia';
export const useAlertStore = defineStore('alert', {
    state: () => ({
        events: [],
        blinkingDeviceIds: new Set(),
    }),
    getters: {
        activeEvents(state) {
            return state.events.filter(e => !e.resolved);
        },
    },
    actions: {
        triggerAlert(event) {
            this.events.unshift(event);
            this.blinkingDeviceIds.add(event.deviceId);
        },
        resolveAlert(eventId) {
            const event = this.events.find(e => e.id === eventId);
            if (event) {
                event.resolved = true;
                // Stop blinking if no other active events for the device
                const deviceHasActive = this.events.some(e => e.deviceId === event.deviceId && !e.resolved);
                if (!deviceHasActive)
                    this.blinkingDeviceIds.delete(event.deviceId);
            }
        },
        clearBlink(deviceId) {
            this.blinkingDeviceIds.delete(deviceId);
        },
    },
});
