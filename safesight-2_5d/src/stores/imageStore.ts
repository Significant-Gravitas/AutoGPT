import { defineStore } from 'pinia'

export type DeviceMarker = {
  id: string
  name: string
  x: number // normalized [0,1] image coordinates
  y: number // normalized [0,1]
  status?: 'normal' | 'warning' | 'alarm'
  meta?: Record<string, unknown>
}

export type TiledImageInfo = {
  id: string
  name: string
  width: number
  height: number
  tileSize: number
  tileOverlap: number
  levels: number
  // Preferred runtime tile source (with getTileUrl)
  tileSource?: any
  // Optional legacy DZI url
  dziUrl?: string
}

export const useImageStore = defineStore('image', {
  state: () => ({
    image: null as TiledImageInfo | null,
    markers: [] as DeviceMarker[],
    selectedMarkerId: null as string | null,
    pendingMarker: null as null | { id: string; name: string },
  }),
  getters: {
    selectedMarker(state) {
      return state.markers.find(m => m.id === state.selectedMarkerId) || null
    },
  },
  actions: {
    setImage(info: TiledImageInfo | null) {
      this.image = info
      this.markers = []
      this.selectedMarkerId = null
    },
    addMarker(marker: DeviceMarker) {
      this.markers.push(marker)
    },
    startPlacingMarker(id: string, name: string) {
      this.pendingMarker = { id, name }
    },
    cancelPlacingMarker() {
      this.pendingMarker = null
    },
    completePlacingMarker(xNormalized: number, yNormalized: number) {
      if (!this.pendingMarker) return
      const { id, name } = this.pendingMarker
      this.markers.push({ id, name, x: xNormalized, y: yNormalized, status: 'normal' })
      this.pendingMarker = null
    },
    updateMarker(id: string, patch: Partial<DeviceMarker>) {
      const idx = this.markers.findIndex(m => m.id === id)
      if (idx !== -1) {
        this.markers[idx] = { ...this.markers[idx], ...patch }
      }
    },
    removeMarker(id: string) {
      this.markers = this.markers.filter(m => m.id !== id)
      if (this.selectedMarkerId === id) this.selectedMarkerId = null
    },
    selectMarker(id: string | null) {
      this.selectedMarkerId = id
    },
  },
})