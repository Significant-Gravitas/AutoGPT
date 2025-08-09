import { onMounted, onBeforeUnmount, ref, watch, nextTick } from 'vue';
import * as OpenSeadragon from 'openseadragon';
import { useImageStore } from '../stores/imageStore';
import { useAlertStore } from '../stores/alertStore';
const imageStore = useImageStore();
const alertStore = useAlertStore();
const osdEl = ref(null);
let viewer = null;
const markerElements = new Map();
function destroyViewer() {
    if (viewer) {
        try {
            viewer.destroy();
        }
        catch { }
        viewer = null;
        markerElements.clear();
    }
}
function initViewer() {
    destroyViewer();
    if (!osdEl.value)
        return;
    viewer = OpenSeadragon({
        element: osdEl.value,
        crossOriginPolicy: 'Anonymous',
        prefixUrl: 'https://openseadragon.github.io/openseadragon/images/',
        showNavigator: true,
        animationTime: 0.8,
        maxZoomPixelRatio: 2,
        visibilityRatio: 1,
        constrainDuringPan: true,
    });
    viewer.addHandler('canvas-click', (ev) => {
        if (!viewer)
            return;
        if (!imageStore.image)
            return;
        if (!ev.quick)
            return;
        const webPoint = ev.position;
        const viewportPoint = viewer.viewport.pointFromPixel(webPoint);
        const imagePoint = viewer.viewport.viewportToImageCoordinates(viewportPoint);
        if (imageStore.pendingMarker) {
            const xNorm = imagePoint.x / imageStore.image.width;
            const yNorm = imagePoint.y / imageStore.image.height;
            imageStore.completePlacingMarker(xNorm, yNorm);
            renderMarkers();
            return;
        }
    });
    if (imageStore.image?.dziUrl) {
        viewer.open(imageStore.image.dziUrl);
        viewer.addOnceHandler('open', () => {
            renderMarkers();
        });
    }
}
function renderMarkers() {
    if (!viewer)
        return;
    // Clear previous overlays
    markerElements.forEach((el, id) => {
        try {
            viewer.removeOverlay(el);
        }
        catch { }
    });
    markerElements.clear();
    const img = imageStore.image;
    if (!img)
        return;
    for (const m of imageStore.markers) {
        const el = document.createElement('div');
        el.className = 'marker';
        el.dataset.id = m.id;
        el.innerHTML = `<span class="dot"></span><span class="label">${m.name}</span>`;
        el.onclick = () => imageStore.selectMarker(m.id);
        // Blink if alarm
        const shouldBlink = alertStore.blinkingDeviceIds.has(m.id);
        if (shouldBlink)
            el.classList.add('blinking');
        const imageX = m.x * img.width;
        const imageY = m.y * img.height;
        const vpPoint = viewer.viewport.imageToViewportCoordinates(imageX, imageY);
        viewer.addOverlay({ element: el, location: new OpenSeadragon.Point(vpPoint.x, vpPoint.y), placement: OpenSeadragon.Placement.CENTER });
        markerElements.set(m.id, el);
    }
}
// watch for data changes
watch(() => imageStore.image?.dziUrl, async () => {
    await nextTick();
    initViewer();
});
watch(() => imageStore.markers.slice(), () => {
    renderMarkers();
}, { deep: true });
watch(() => alertStore.blinkingDeviceIds.size, () => {
    // update blink classes
    markerElements.forEach((el, id) => {
        if (alertStore.blinkingDeviceIds.has(id))
            el.classList.add('blinking');
        else
            el.classList.remove('blinking');
    });
});
onMounted(() => {
    initViewer();
});
onBeforeUnmount(() => {
    destroyViewer();
});
debugger; /* PartiallyEnd: #3632/scriptSetup.vue */
const __VLS_ctx = {};
let __VLS_elements;
let __VLS_components;
let __VLS_directives;
/** @type {__VLS_StyleScopedClasses['marker']} */ ;
/** @type {__VLS_StyleScopedClasses['marker']} */ ;
/** @type {__VLS_StyleScopedClasses['marker']} */ ;
/** @type {__VLS_StyleScopedClasses['dot']} */ ;
// CSS variable injection 
// CSS variable injection end 
__VLS_asFunctionalElement(__VLS_elements.div, __VLS_elements.div)({
    ...{ class: "viewer-root" },
});
__VLS_asFunctionalElement(__VLS_elements.div, __VLS_elements.div)({
    ref: "osdEl",
    ...{ class: "osd-container" },
});
/** @type {typeof __VLS_ctx.osdEl} */ ;
// @ts-ignore
[osdEl,];
/** @type {__VLS_StyleScopedClasses['viewer-root']} */ ;
/** @type {__VLS_StyleScopedClasses['osd-container']} */ ;
var __VLS_dollars;
const __VLS_self = (await import('vue')).defineComponent({
    setup() {
        return {
            osdEl: osdEl,
        };
    },
});
export default (await import('vue')).defineComponent({
    setup() {
    },
});
; /* PartiallyEnd: #4569/main.vue */
