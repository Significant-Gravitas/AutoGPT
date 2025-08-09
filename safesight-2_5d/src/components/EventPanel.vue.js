import { useAlertStore } from '../stores/alertStore';
import { useImageStore } from '../stores/imageStore';
const alertStore = useAlertStore();
const imageStore = useImageStore();
function resolve(id) {
    alertStore.resolveAlert(id);
}
function focusDevice(deviceId) {
    imageStore.selectMarker(deviceId);
}
debugger; /* PartiallyEnd: #3632/scriptSetup.vue */
const __VLS_ctx = {};
let __VLS_elements;
let __VLS_components;
let __VLS_directives;
const __VLS_0 = {}.ElCard;
/** @type {[typeof __VLS_components.ElCard, typeof __VLS_components.elCard, typeof __VLS_components.ElCard, typeof __VLS_components.elCard, ]} */ ;
// @ts-ignore
ElCard;
// @ts-ignore
const __VLS_1 = __VLS_asFunctionalComponent(__VLS_0, new __VLS_0({
    shadow: "never",
    header: "事件预警",
}));
const __VLS_2 = __VLS_1({
    shadow: "never",
    header: "事件预警",
}, ...__VLS_functionalComponentArgsRest(__VLS_1));
var __VLS_4 = {};
const { default: __VLS_5 } = __VLS_3.slots;
const __VLS_6 = {}.ElTable;
/** @type {[typeof __VLS_components.ElTable, typeof __VLS_components.elTable, typeof __VLS_components.ElTable, typeof __VLS_components.elTable, ]} */ ;
// @ts-ignore
ElTable;
// @ts-ignore
const __VLS_7 = __VLS_asFunctionalComponent(__VLS_6, new __VLS_6({
    data: (__VLS_ctx.alertStore.events),
    size: "small",
    ...{ style: {} },
}));
const __VLS_8 = __VLS_7({
    data: (__VLS_ctx.alertStore.events),
    size: "small",
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_7));
const { default: __VLS_10 } = __VLS_9.slots;
// @ts-ignore
[alertStore,];
const __VLS_11 = {}.ElTableColumn;
/** @type {[typeof __VLS_components.ElTableColumn, typeof __VLS_components.elTableColumn, ]} */ ;
// @ts-ignore
ElTableColumn;
// @ts-ignore
const __VLS_12 = __VLS_asFunctionalComponent(__VLS_11, new __VLS_11({
    prop: "timestamp",
    label: "时间",
    width: "170",
}));
const __VLS_13 = __VLS_12({
    prop: "timestamp",
    label: "时间",
    width: "170",
}, ...__VLS_functionalComponentArgsRest(__VLS_12));
const __VLS_16 = {}.ElTableColumn;
/** @type {[typeof __VLS_components.ElTableColumn, typeof __VLS_components.elTableColumn, ]} */ ;
// @ts-ignore
ElTableColumn;
// @ts-ignore
const __VLS_17 = __VLS_asFunctionalComponent(__VLS_16, new __VLS_16({
    prop: "deviceId",
    label: "设备ID",
    width: "120",
}));
const __VLS_18 = __VLS_17({
    prop: "deviceId",
    label: "设备ID",
    width: "120",
}, ...__VLS_functionalComponentArgsRest(__VLS_17));
const __VLS_21 = {}.ElTableColumn;
/** @type {[typeof __VLS_components.ElTableColumn, typeof __VLS_components.elTableColumn, ]} */ ;
// @ts-ignore
ElTableColumn;
// @ts-ignore
const __VLS_22 = __VLS_asFunctionalComponent(__VLS_21, new __VLS_21({
    prop: "type",
    label: "类型",
    width: "100",
}));
const __VLS_23 = __VLS_22({
    prop: "type",
    label: "类型",
    width: "100",
}, ...__VLS_functionalComponentArgsRest(__VLS_22));
const __VLS_26 = {}.ElTableColumn;
/** @type {[typeof __VLS_components.ElTableColumn, typeof __VLS_components.elTableColumn, ]} */ ;
// @ts-ignore
ElTableColumn;
// @ts-ignore
const __VLS_27 = __VLS_asFunctionalComponent(__VLS_26, new __VLS_26({
    prop: "level",
    label: "等级",
    width: "80",
}));
const __VLS_28 = __VLS_27({
    prop: "level",
    label: "等级",
    width: "80",
}, ...__VLS_functionalComponentArgsRest(__VLS_27));
const __VLS_31 = {}.ElTableColumn;
/** @type {[typeof __VLS_components.ElTableColumn, typeof __VLS_components.elTableColumn, typeof __VLS_components.ElTableColumn, typeof __VLS_components.elTableColumn, ]} */ ;
// @ts-ignore
ElTableColumn;
// @ts-ignore
const __VLS_32 = __VLS_asFunctionalComponent(__VLS_31, new __VLS_31({
    label: "描述",
}));
const __VLS_33 = __VLS_32({
    label: "描述",
}, ...__VLS_functionalComponentArgsRest(__VLS_32));
const { default: __VLS_35 } = __VLS_34.slots;
{
    const { default: __VLS_36 } = __VLS_34.slots;
    const [{ row }] = __VLS_getSlotParameters(__VLS_36);
    (row.description);
}
var __VLS_34;
const __VLS_37 = {}.ElTableColumn;
/** @type {[typeof __VLS_components.ElTableColumn, typeof __VLS_components.elTableColumn, typeof __VLS_components.ElTableColumn, typeof __VLS_components.elTableColumn, ]} */ ;
// @ts-ignore
ElTableColumn;
// @ts-ignore
const __VLS_38 = __VLS_asFunctionalComponent(__VLS_37, new __VLS_37({
    label: "状态",
    width: "80",
}));
const __VLS_39 = __VLS_38({
    label: "状态",
    width: "80",
}, ...__VLS_functionalComponentArgsRest(__VLS_38));
const { default: __VLS_41 } = __VLS_40.slots;
{
    const { default: __VLS_42 } = __VLS_40.slots;
    const [{ row }] = __VLS_getSlotParameters(__VLS_42);
    const __VLS_43 = {}.ElTag;
    /** @type {[typeof __VLS_components.ElTag, typeof __VLS_components.elTag, typeof __VLS_components.ElTag, typeof __VLS_components.elTag, ]} */ ;
    // @ts-ignore
    ElTag;
    // @ts-ignore
    const __VLS_44 = __VLS_asFunctionalComponent(__VLS_43, new __VLS_43({
        type: (row.resolved ? 'info' : 'danger'),
    }));
    const __VLS_45 = __VLS_44({
        type: (row.resolved ? 'info' : 'danger'),
    }, ...__VLS_functionalComponentArgsRest(__VLS_44));
    const { default: __VLS_47 } = __VLS_46.slots;
    (row.resolved ? '已处理' : '未处理');
    var __VLS_46;
}
var __VLS_40;
const __VLS_48 = {}.ElTableColumn;
/** @type {[typeof __VLS_components.ElTableColumn, typeof __VLS_components.elTableColumn, typeof __VLS_components.ElTableColumn, typeof __VLS_components.elTableColumn, ]} */ ;
// @ts-ignore
ElTableColumn;
// @ts-ignore
const __VLS_49 = __VLS_asFunctionalComponent(__VLS_48, new __VLS_48({
    label: "操作",
    width: "180",
}));
const __VLS_50 = __VLS_49({
    label: "操作",
    width: "180",
}, ...__VLS_functionalComponentArgsRest(__VLS_49));
const { default: __VLS_52 } = __VLS_51.slots;
{
    const { default: __VLS_53 } = __VLS_51.slots;
    const [{ row }] = __VLS_getSlotParameters(__VLS_53);
    const __VLS_54 = {}.ElButton;
    /** @type {[typeof __VLS_components.ElButton, typeof __VLS_components.elButton, typeof __VLS_components.ElButton, typeof __VLS_components.elButton, ]} */ ;
    // @ts-ignore
    ElButton;
    // @ts-ignore
    const __VLS_55 = __VLS_asFunctionalComponent(__VLS_54, new __VLS_54({
        ...{ 'onClick': {} },
        size: "small",
    }));
    const __VLS_56 = __VLS_55({
        ...{ 'onClick': {} },
        size: "small",
    }, ...__VLS_functionalComponentArgsRest(__VLS_55));
    let __VLS_58;
    let __VLS_59;
    const __VLS_60 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.focusDevice(row.deviceId);
                // @ts-ignore
                [focusDevice,];
            } });
    const { default: __VLS_61 } = __VLS_57.slots;
    var __VLS_57;
    const __VLS_62 = {}.ElButton;
    /** @type {[typeof __VLS_components.ElButton, typeof __VLS_components.elButton, typeof __VLS_components.ElButton, typeof __VLS_components.elButton, ]} */ ;
    // @ts-ignore
    ElButton;
    // @ts-ignore
    const __VLS_63 = __VLS_asFunctionalComponent(__VLS_62, new __VLS_62({
        ...{ 'onClick': {} },
        size: "small",
        type: "success",
        disabled: (row.resolved),
    }));
    const __VLS_64 = __VLS_63({
        ...{ 'onClick': {} },
        size: "small",
        type: "success",
        disabled: (row.resolved),
    }, ...__VLS_functionalComponentArgsRest(__VLS_63));
    let __VLS_66;
    let __VLS_67;
    const __VLS_68 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.resolve(row.id);
                // @ts-ignore
                [resolve,];
            } });
    const { default: __VLS_69 } = __VLS_65.slots;
    var __VLS_65;
}
var __VLS_51;
var __VLS_9;
var __VLS_3;
var __VLS_dollars;
const __VLS_self = (await import('vue')).defineComponent({
    setup() {
        return {
            alertStore: alertStore,
            resolve: resolve,
            focusDevice: focusDevice,
        };
    },
});
export default (await import('vue')).defineComponent({
    setup() {
    },
});
; /* PartiallyEnd: #4569/main.vue */
