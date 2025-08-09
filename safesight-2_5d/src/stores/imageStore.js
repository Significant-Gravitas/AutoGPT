import { defineStore } from 'pinia';
export const useImageStore = defineStore('image', {
    state: () => ({
        image: null,
        markers: [],
        selectedMarkerId: null,
        pendingMarker: null,
    }),
    getters: {
        selectedMarker(state) {
            return state.markers.find(m => m.id === state.selectedMarkerId) || null;
        },
    },
    actions: {
        setImage(info) {
            this.image = info;
            this.markers = [];
            this.selectedMarkerId = null;
        },
        addMarker(marker) {
            this.markers.push(marker);
        },
        startPlacingMarker(id, name) {
            this.pendingMarker = { id, name };
        },
        cancelPlacingMarker() {
            this.pendingMarker = null;
        },
        completePlacingMarker(xNormalized, yNormalized) {
            if (!this.pendingMarker)
                return;
            const { id, name } = this.pendingMarker;
            this.markers.push({ id, name, x: xNormalized, y: yNormalized, status: 'normal' });
            this.pendingMarker = null;
        },
        updateMarker(id, patch) {
            const idx = this.markers.findIndex(m => m.id === id);
            if (idx !== -1) {
                this.markers[idx] = { ...this.markers[idx], ...patch };
            }
        },
        removeMarker(id) {
            this.markers = this.markers.filter(m => m.id !== id);
            if (this.selectedMarkerId === id)
                this.selectedMarkerId = null;
        },
        selectMarker(id) {
            this.selectedMarkerId = id;
        },
    },
});
