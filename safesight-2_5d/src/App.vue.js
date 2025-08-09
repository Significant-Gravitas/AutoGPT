import UploadImage from './components/UploadImage.vue';
import DeepZoomViewer from './components/DeepZoomViewer.vue';
import DevicePanel from './components/DevicePanel.vue';
import EventPanel from './components/EventPanel.vue';
import StrategyPanel from './components/StrategyPanel.vue';
import { useAlertStore } from './stores/alertStore';
const alertStore = useAlertStore();
function simulateAlert() {
    alertStore.triggerAlert({
        id: 'evt-' + Date.now(),
        deviceId: 'device-1',
        type: '告警',
        level: '高',
        timestamp: new Date().toISOString(),
        description: '区域入侵预警',
    });
}
debugger; /* PartiallyEnd: #3632/scriptSetup.vue */
const __VLS_ctx = {};
let __VLS_elements;
let __VLS_components;
let __VLS_directives;
// CSS variable injection 
// CSS variable injection end 
__VLS_asFunctionalElement(__VLS_elements.div, __VLS_elements.div)({
    ...{ class: "app-container" },
});
__VLS_asFunctionalElement(__VLS_elements.header, __VLS_elements.header)({
    ...{ class: "app-header" },
});
__VLS_asFunctionalElement(__VLS_elements.h1, __VLS_elements.h1)({});
__VLS_asFunctionalElement(__VLS_elements.div, __VLS_elements.div)({
    ...{ class: "header-actions" },
});
/** @type {[typeof UploadImage, ]} */ ;
// @ts-ignore
const __VLS_0 = __VLS_asFunctionalComponent(UploadImage, new UploadImage({}));
const __VLS_1 = __VLS_0({}, ...__VLS_functionalComponentArgsRest(__VLS_0));
const __VLS_4 = {}.ElButton;
/** @type {[typeof __VLS_components.ElButton, typeof __VLS_components.elButton, typeof __VLS_components.ElButton, typeof __VLS_components.elButton, ]} */ ;
// @ts-ignore
ElButton;
// @ts-ignore
const __VLS_5 = __VLS_asFunctionalComponent(__VLS_4, new __VLS_4({
    ...{ 'onClick': {} },
    type: "primary",
}));
const __VLS_6 = __VLS_5({
    ...{ 'onClick': {} },
    type: "primary",
}, ...__VLS_functionalComponentArgsRest(__VLS_5));
let __VLS_8;
let __VLS_9;
const __VLS_10 = ({ click: {} },
    { onClick: (__VLS_ctx.simulateAlert) });
const { default: __VLS_11 } = __VLS_7.slots;
// @ts-ignore
[simulateAlert,];
var __VLS_7;
__VLS_asFunctionalElement(__VLS_elements.main, __VLS_elements.main)({
    ...{ class: "app-main" },
});
__VLS_asFunctionalElement(__VLS_elements.section, __VLS_elements.section)({
    ...{ class: "viewer-section" },
});
/** @type {[typeof DeepZoomViewer, ]} */ ;
// @ts-ignore
const __VLS_12 = __VLS_asFunctionalComponent(DeepZoomViewer, new DeepZoomViewer({}));
const __VLS_13 = __VLS_12({}, ...__VLS_functionalComponentArgsRest(__VLS_12));
__VLS_asFunctionalElement(__VLS_elements.aside, __VLS_elements.aside)({
    ...{ class: "side-section" },
});
/** @type {[typeof DevicePanel, ]} */ ;
// @ts-ignore
const __VLS_16 = __VLS_asFunctionalComponent(DevicePanel, new DevicePanel({}));
const __VLS_17 = __VLS_16({}, ...__VLS_functionalComponentArgsRest(__VLS_16));
/** @type {[typeof EventPanel, ]} */ ;
// @ts-ignore
const __VLS_20 = __VLS_asFunctionalComponent(EventPanel, new EventPanel({}));
const __VLS_21 = __VLS_20({}, ...__VLS_functionalComponentArgsRest(__VLS_20));
/** @type {[typeof StrategyPanel, ]} */ ;
// @ts-ignore
const __VLS_24 = __VLS_asFunctionalComponent(StrategyPanel, new StrategyPanel({}));
const __VLS_25 = __VLS_24({}, ...__VLS_functionalComponentArgsRest(__VLS_24));
/** @type {__VLS_StyleScopedClasses['app-container']} */ ;
/** @type {__VLS_StyleScopedClasses['app-header']} */ ;
/** @type {__VLS_StyleScopedClasses['header-actions']} */ ;
/** @type {__VLS_StyleScopedClasses['app-main']} */ ;
/** @type {__VLS_StyleScopedClasses['viewer-section']} */ ;
/** @type {__VLS_StyleScopedClasses['side-section']} */ ;
var __VLS_dollars;
const __VLS_self = (await import('vue')).defineComponent({
    setup() {
        return {
            UploadImage: UploadImage,
            DeepZoomViewer: DeepZoomViewer,
            DevicePanel: DevicePanel,
            EventPanel: EventPanel,
            StrategyPanel: StrategyPanel,
            simulateAlert: simulateAlert,
        };
    },
});
export default (await import('vue')).defineComponent({
    setup() {
    },
});
; /* PartiallyEnd: #4569/main.vue */
