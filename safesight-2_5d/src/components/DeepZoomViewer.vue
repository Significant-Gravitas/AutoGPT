<template>
  <div class="viewer-root">
    <div ref="osdEl" class="osd-container"></div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, onBeforeUnmount, ref, watch, nextTick } from 'vue'
import * as OpenSeadragon from 'openseadragon'
import { useImageStore } from '../stores/imageStore'
import { useAlertStore } from '../stores/alertStore'

const imageStore = useImageStore()
const alertStore = useAlertStore()

const osdEl = ref<HTMLDivElement | null>(null)
let viewer: OpenSeadragon.Viewer | null = null

const markerElements = new Map<string, HTMLElement>()

function destroyViewer() {
  if (viewer) {
    try { viewer.destroy() } catch {}
    viewer = null
    markerElements.clear()
  }
}

function initViewer() {
  destroyViewer()
  if (!osdEl.value) return
  viewer = new OpenSeadragon.Viewer({
    element: osdEl.value as any,
    crossOriginPolicy: 'Anonymous' as any,
    prefixUrl: 'https://openseadragon.github.io/openseadragon/images/',
    showNavigator: true,
    animationTime: 0.8,
    maxZoomPixelRatio: 2,
    visibilityRatio: 1,
    constrainDuringPan: true,
  })

  viewer.addHandler('canvas-click', (ev: any) => {
    if (!viewer) return
    if (!imageStore.image) return
    if (!ev.quick) return

    const webPoint = ev.position as OpenSeadragon.Point
    const viewportPoint = viewer.viewport.pointFromPixel(webPoint)
    const imagePoint = viewer.viewport.viewportToImageCoordinates(viewportPoint)

    if (imageStore.pendingMarker) {
      const xNorm = imagePoint.x / imageStore.image.width
      const yNorm = imagePoint.y / imageStore.image.height
      imageStore.completePlacingMarker(xNorm, yNorm)
      renderMarkers()
      return
    }
  })

  if (imageStore.image?.dziUrl) {
    viewer.open(imageStore.image.dziUrl)
    viewer.addOnceHandler('open', () => {
      renderMarkers()
    })
  }
}

function renderMarkers() {
  if (!viewer) return
  // Clear previous overlays
  markerElements.forEach((el) => {
    try { viewer!.removeOverlay(el) } catch {}
  })
  markerElements.clear()

  const img = imageStore.image
  if (!img) return

  for (const m of imageStore.markers) {
    const el = document.createElement('div')
    el.className = 'marker'
    el.dataset.id = m.id
    el.innerHTML = `<span class="dot"></span><span class="label">${m.name}</span>`
    el.onclick = () => imageStore.selectMarker(m.id)

    // Blink if alarm
    const shouldBlink = alertStore.blinkingDeviceIds.has(m.id)
    if (shouldBlink) el.classList.add('blinking')

    const imageX = m.x * img.width
    const imageY = m.y * img.height
    const vpPoint = viewer.viewport.imageToViewportCoordinates(imageX, imageY)
    viewer.addOverlay({ element: el, location: new OpenSeadragon.Point(vpPoint.x, vpPoint.y), placement: OpenSeadragon.Placement.CENTER })
    markerElements.set(m.id, el)
  }
}

// watch for data changes
watch(() => imageStore.image?.dziUrl, async () => {
  await nextTick()
  initViewer()
})

watch(() => imageStore.markers.slice(), () => {
  renderMarkers()
}, { deep: true })

watch(() => alertStore.blinkingDeviceIds.size, () => {
  // update blink classes
  markerElements.forEach((el, id) => {
    if (alertStore.blinkingDeviceIds.has(id)) el.classList.add('blinking')
    else el.classList.remove('blinking')
  })
})

onMounted(() => {
  initViewer()
})

onBeforeUnmount(() => {
  destroyViewer()
})
</script>

<style scoped>
.viewer-root {
  position: absolute;
  inset: 0;
}
.osd-container {
  position: absolute;
  inset: 0;
}
.marker {
  position: absolute;
  transform: translate(-50%, -100%);
  display: flex;
  align-items: center;
  gap: 6px;
  pointer-events: auto;
}
.marker .dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: #22c55e;
  box-shadow: 0 0 0 2px rgba(34,197,94,0.25);
}
.marker .label {
  color: #e5e7eb;
  background: rgba(17,24,39,0.8);
  border: 1px solid rgba(255,255,255,0.15);
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 12px;
  white-space: nowrap;
}
.marker.blinking .dot {
  background: #ef4444;
  animation: blink 1s infinite;
}
@keyframes blink {
  0%, 100% { box-shadow: 0 0 0 2px rgba(239,68,68,0.25); opacity: 1; }
  50% { box-shadow: 0 0 0 6px rgba(239,68,68,0.4); opacity: 0.5; }
}
</style>