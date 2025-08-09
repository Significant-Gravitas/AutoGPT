import { reactive } from 'vue';
import { useImageStore } from '../stores/imageStore';
const imageStore = useImageStore();
const form = reactive({ id: '', name: '' });
function startPlace() {
    imageStore.startPlacingMarker(form.id.trim(), form.name.trim());
}
function cancelPlace() {
    imageStore.cancelPlacingMarker();
}
function select(id) {
    imageStore.selectMarker(id);
}
function remove(id) {
    imageStore.removeMarker(id);
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
    header: "设备管理",
}));
const __VLS_2 = __VLS_1({
    shadow: "never",
    header: "设备管理",
}, ...__VLS_functionalComponentArgsRest(__VLS_1));
var __VLS_4 = {};
const { default: __VLS_5 } = __VLS_3.slots;
const __VLS_6 = {}.ElForm;
/** @type {[typeof __VLS_components.ElForm, typeof __VLS_components.elForm, typeof __VLS_components.ElForm, typeof __VLS_components.elForm, ]} */ ;
// @ts-ignore
ElForm;
// @ts-ignore
const __VLS_7 = __VLS_asFunctionalComponent(__VLS_6, new __VLS_6({
    ...{ 'onSubmit': {} },
    inline: (true),
}));
const __VLS_8 = __VLS_7({
    ...{ 'onSubmit': {} },
    inline: (true),
}, ...__VLS_functionalComponentArgsRest(__VLS_7));
let __VLS_10;
let __VLS_11;
const __VLS_12 = ({ submit: {} },
    { onSubmit: () => { } });
const { default: __VLS_13 } = __VLS_9.slots;
const __VLS_14 = {}.ElFormItem;
/** @type {[typeof __VLS_components.ElFormItem, typeof __VLS_components.elFormItem, typeof __VLS_components.ElFormItem, typeof __VLS_components.elFormItem, ]} */ ;
// @ts-ignore
ElFormItem;
// @ts-ignore
const __VLS_15 = __VLS_asFunctionalComponent(__VLS_14, new __VLS_14({
    label: "设备ID",
}));
const __VLS_16 = __VLS_15({
    label: "设备ID",
}, ...__VLS_functionalComponentArgsRest(__VLS_15));
const { default: __VLS_18 } = __VLS_17.slots;
const __VLS_19 = {}.ElInput;
/** @type {[typeof __VLS_components.ElInput, typeof __VLS_components.elInput, ]} */ ;
// @ts-ignore
ElInput;
// @ts-ignore
const __VLS_20 = __VLS_asFunctionalComponent(__VLS_19, new __VLS_19({
    modelValue: (__VLS_ctx.form.id),
    placeholder: "如 device-1",
    ...{ style: {} },
}));
const __VLS_21 = __VLS_20({
    modelValue: (__VLS_ctx.form.id),
    placeholder: "如 device-1",
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_20));
// @ts-ignore
[form,];
var __VLS_17;
const __VLS_24 = {}.ElFormItem;
/** @type {[typeof __VLS_components.ElFormItem, typeof __VLS_components.elFormItem, typeof __VLS_components.ElFormItem, typeof __VLS_components.elFormItem, ]} */ ;
// @ts-ignore
ElFormItem;
// @ts-ignore
const __VLS_25 = __VLS_asFunctionalComponent(__VLS_24, new __VLS_24({
    label: "名称",
}));
const __VLS_26 = __VLS_25({
    label: "名称",
}, ...__VLS_functionalComponentArgsRest(__VLS_25));
const { default: __VLS_28 } = __VLS_27.slots;
const __VLS_29 = {}.ElInput;
/** @type {[typeof __VLS_components.ElInput, typeof __VLS_components.elInput, ]} */ ;
// @ts-ignore
ElInput;
// @ts-ignore
const __VLS_30 = __VLS_asFunctionalComponent(__VLS_29, new __VLS_29({
    modelValue: (__VLS_ctx.form.name),
    placeholder: "摄像头A",
    ...{ style: {} },
}));
const __VLS_31 = __VLS_30({
    modelValue: (__VLS_ctx.form.name),
    placeholder: "摄像头A",
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_30));
// @ts-ignore
[form,];
var __VLS_27;
const __VLS_34 = {}.ElFormItem;
/** @type {[typeof __VLS_components.ElFormItem, typeof __VLS_components.elFormItem, typeof __VLS_components.ElFormItem, typeof __VLS_components.elFormItem, ]} */ ;
// @ts-ignore
ElFormItem;
// @ts-ignore
const __VLS_35 = __VLS_asFunctionalComponent(__VLS_34, new __VLS_34({}));
const __VLS_36 = __VLS_35({}, ...__VLS_functionalComponentArgsRest(__VLS_35));
const { default: __VLS_38 } = __VLS_37.slots;
const __VLS_39 = {}.ElButton;
/** @type {[typeof __VLS_components.ElButton, typeof __VLS_components.elButton, typeof __VLS_components.ElButton, typeof __VLS_components.elButton, ]} */ ;
// @ts-ignore
ElButton;
// @ts-ignore
const __VLS_40 = __VLS_asFunctionalComponent(__VLS_39, new __VLS_39({
    ...{ 'onClick': {} },
    type: "primary",
    disabled: (!__VLS_ctx.form.id || !__VLS_ctx.form.name),
}));
const __VLS_41 = __VLS_40({
    ...{ 'onClick': {} },
    type: "primary",
    disabled: (!__VLS_ctx.form.id || !__VLS_ctx.form.name),
}, ...__VLS_functionalComponentArgsRest(__VLS_40));
let __VLS_43;
let __VLS_44;
const __VLS_45 = ({ click: {} },
    { onClick: (__VLS_ctx.startPlace) });
const { default: __VLS_46 } = __VLS_42.slots;
// @ts-ignore
[form, form, startPlace,];
var __VLS_42;
const __VLS_47 = {}.ElButton;
/** @type {[typeof __VLS_components.ElButton, typeof __VLS_components.elButton, typeof __VLS_components.ElButton, typeof __VLS_components.elButton, ]} */ ;
// @ts-ignore
ElButton;
// @ts-ignore
const __VLS_48 = __VLS_asFunctionalComponent(__VLS_47, new __VLS_47({
    ...{ 'onClick': {} },
    disabled: (!__VLS_ctx.imageStore.pendingMarker),
}));
const __VLS_49 = __VLS_48({
    ...{ 'onClick': {} },
    disabled: (!__VLS_ctx.imageStore.pendingMarker),
}, ...__VLS_functionalComponentArgsRest(__VLS_48));
let __VLS_51;
let __VLS_52;
const __VLS_53 = ({ click: {} },
    { onClick: (__VLS_ctx.cancelPlace) });
const { default: __VLS_54 } = __VLS_50.slots;
// @ts-ignore
[imageStore, cancelPlace,];
var __VLS_50;
var __VLS_37;
var __VLS_9;
const __VLS_55 = {}.ElTable;
/** @type {[typeof __VLS_components.ElTable, typeof __VLS_components.elTable, typeof __VLS_components.ElTable, typeof __VLS_components.elTable, ]} */ ;
// @ts-ignore
ElTable;
// @ts-ignore
const __VLS_56 = __VLS_asFunctionalComponent(__VLS_55, new __VLS_55({
    data: (__VLS_ctx.imageStore.markers),
    size: "small",
    ...{ style: {} },
}));
const __VLS_57 = __VLS_56({
    data: (__VLS_ctx.imageStore.markers),
    size: "small",
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_56));
const { default: __VLS_59 } = __VLS_58.slots;
// @ts-ignore
[imageStore,];
const __VLS_60 = {}.ElTableColumn;
/** @type {[typeof __VLS_components.ElTableColumn, typeof __VLS_components.elTableColumn, ]} */ ;
// @ts-ignore
ElTableColumn;
// @ts-ignore
const __VLS_61 = __VLS_asFunctionalComponent(__VLS_60, new __VLS_60({
    prop: "id",
    label: "ID",
    width: "120",
}));
const __VLS_62 = __VLS_61({
    prop: "id",
    label: "ID",
    width: "120",
}, ...__VLS_functionalComponentArgsRest(__VLS_61));
const __VLS_65 = {}.ElTableColumn;
/** @type {[typeof __VLS_components.ElTableColumn, typeof __VLS_components.elTableColumn, ]} */ ;
// @ts-ignore
ElTableColumn;
// @ts-ignore
const __VLS_66 = __VLS_asFunctionalComponent(__VLS_65, new __VLS_65({
    prop: "name",
    label: "名称",
}));
const __VLS_67 = __VLS_66({
    prop: "name",
    label: "名称",
}, ...__VLS_functionalComponentArgsRest(__VLS_66));
const __VLS_70 = {}.ElTableColumn;
/** @type {[typeof __VLS_components.ElTableColumn, typeof __VLS_components.elTableColumn, typeof __VLS_components.ElTableColumn, typeof __VLS_components.elTableColumn, ]} */ ;
// @ts-ignore
ElTableColumn;
// @ts-ignore
const __VLS_71 = __VLS_asFunctionalComponent(__VLS_70, new __VLS_70({
    label: "坐标",
    width: "140",
}));
const __VLS_72 = __VLS_71({
    label: "坐标",
    width: "140",
}, ...__VLS_functionalComponentArgsRest(__VLS_71));
const { default: __VLS_74 } = __VLS_73.slots;
{
    const { default: __VLS_75 } = __VLS_73.slots;
    const [{ row }] = __VLS_getSlotParameters(__VLS_75);
    (row.x.toFixed(3));
    (row.y.toFixed(3));
}
var __VLS_73;
const __VLS_76 = {}.ElTableColumn;
/** @type {[typeof __VLS_components.ElTableColumn, typeof __VLS_components.elTableColumn, typeof __VLS_components.ElTableColumn, typeof __VLS_components.elTableColumn, ]} */ ;
// @ts-ignore
ElTableColumn;
// @ts-ignore
const __VLS_77 = __VLS_asFunctionalComponent(__VLS_76, new __VLS_76({
    label: "操作",
    width: "160",
}));
const __VLS_78 = __VLS_77({
    label: "操作",
    width: "160",
}, ...__VLS_functionalComponentArgsRest(__VLS_77));
const { default: __VLS_80 } = __VLS_79.slots;
{
    const { default: __VLS_81 } = __VLS_79.slots;
    const [{ row }] = __VLS_getSlotParameters(__VLS_81);
    const __VLS_82 = {}.ElButton;
    /** @type {[typeof __VLS_components.ElButton, typeof __VLS_components.elButton, typeof __VLS_components.ElButton, typeof __VLS_components.elButton, ]} */ ;
    // @ts-ignore
    ElButton;
    // @ts-ignore
    const __VLS_83 = __VLS_asFunctionalComponent(__VLS_82, new __VLS_82({
        ...{ 'onClick': {} },
        size: "small",
    }));
    const __VLS_84 = __VLS_83({
        ...{ 'onClick': {} },
        size: "small",
    }, ...__VLS_functionalComponentArgsRest(__VLS_83));
    let __VLS_86;
    let __VLS_87;
    const __VLS_88 = ({ click: {} },
        { onClick: (...[$event]) => {
                __VLS_ctx.select(row.id);
                // @ts-ignore
                [select,];
            } });
    const { default: __VLS_89 } = __VLS_85.slots;
    var __VLS_85;
    const __VLS_90 = {}.ElPopconfirm;
    /** @type {[typeof __VLS_components.ElPopconfirm, typeof __VLS_components.elPopconfirm, typeof __VLS_components.ElPopconfirm, typeof __VLS_components.elPopconfirm, ]} */ ;
    // @ts-ignore
    ElPopconfirm;
    // @ts-ignore
    const __VLS_91 = __VLS_asFunctionalComponent(__VLS_90, new __VLS_90({
        ...{ 'onConfirm': {} },
        title: "删除该设备？",
    }));
    const __VLS_92 = __VLS_91({
        ...{ 'onConfirm': {} },
        title: "删除该设备？",
    }, ...__VLS_functionalComponentArgsRest(__VLS_91));
    let __VLS_94;
    let __VLS_95;
    const __VLS_96 = ({ confirm: {} },
        { onConfirm: (...[$event]) => {
                __VLS_ctx.remove(row.id);
                // @ts-ignore
                [remove,];
            } });
    const { default: __VLS_97 } = __VLS_93.slots;
    {
        const { reference: __VLS_98 } = __VLS_93.slots;
        const __VLS_99 = {}.ElButton;
        /** @type {[typeof __VLS_components.ElButton, typeof __VLS_components.elButton, typeof __VLS_components.ElButton, typeof __VLS_components.elButton, ]} */ ;
        // @ts-ignore
        ElButton;
        // @ts-ignore
        const __VLS_100 = __VLS_asFunctionalComponent(__VLS_99, new __VLS_99({
            size: "small",
            type: "danger",
        }));
        const __VLS_101 = __VLS_100({
            size: "small",
            type: "danger",
        }, ...__VLS_functionalComponentArgsRest(__VLS_100));
        const { default: __VLS_103 } = __VLS_102.slots;
        var __VLS_102;
    }
    var __VLS_93;
}
var __VLS_79;
var __VLS_58;
var __VLS_3;
var __VLS_dollars;
const __VLS_self = (await import('vue')).defineComponent({
    setup() {
        return {
            imageStore: imageStore,
            form: form,
            startPlace: startPlace,
            cancelPlace: cancelPlace,
            select: select,
            remove: remove,
        };
    },
});
export default (await import('vue')).defineComponent({
    setup() {
    },
});
; /* PartiallyEnd: #4569/main.vue */
