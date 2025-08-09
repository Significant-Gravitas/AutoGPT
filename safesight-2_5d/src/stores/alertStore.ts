import { defineStore } from 'pinia'

export type AlertEvent = {
  id: string
  deviceId: string
  type: string
  level: '低' | '中' | '高'
  timestamp: string
  description?: string
  resolved?: boolean
}

export const useAlertStore = defineStore('alert', {
  state: () => ({
    events: [] as AlertEvent[],
    blinkingDeviceIds: new Set<string>() as Set<string>,
  }),
  getters: {
    activeEvents(state) {
      return state.events.filter(e => !e.resolved)
    },
  },
  actions: {
    triggerAlert(event: AlertEvent) {
      this.events.unshift(event)
      this.blinkingDeviceIds.add(event.deviceId)
    },
    resolveAlert(eventId: string) {
      const event = this.events.find(e => e.id === eventId)
      if (event) {
        event.resolved = true
        // Stop blinking if no other active events for the device
        const deviceHasActive = this.events.some(
          e => e.deviceId === event.deviceId && !e.resolved
        )
        if (!deviceHasActive) this.blinkingDeviceIds.delete(event.deviceId)
      }
    },
    clearBlink(deviceId: string) {
      this.blinkingDeviceIds.delete(deviceId)
    },
  },
})