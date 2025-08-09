import { reactive } from 'vue';
import { useStrategyStore } from '../stores/strategyStore';
const strategyStore = useStrategyStore();
const form = reactive({ name: '', eventType: '', level: '', actions: [] });
function add() {
    strategyStore.addStrategy({
        id: 'stg-' + Date.now(),
        name: form.name,
        condition: { eventType: form.eventType || undefined, level: form.level || undefined },
        actions: form.actions.slice(),
    });
    form.name = '';
    form.eventType = '';
    form.level = '';
    form.actions = [];
}
function remove(id) {
    strategyStore.removeStrategy(id);
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
    header: "联动策略",
}));
const __VLS_2 = __VLS_1({
    shadow: "never",
    header: "联动策略",
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
    label: "名称",
}));
const __VLS_16 = __VLS_15({
    label: "名称",
}, ...__VLS_functionalComponentArgsRest(__VLS_15));
const { default: __VLS_18 } = __VLS_17.slots;
const __VLS_19 = {}.ElInput;
/** @type {[typeof __VLS_components.ElInput, typeof __VLS_components.elInput, ]} */ ;
// @ts-ignore
ElInput;
// @ts-ignore
const __VLS_20 = __VLS_asFunctionalComponent(__VLS_19, new __VLS_19({
    modelValue: (__VLS_ctx.form.name),
    placeholder: "如 入侵-高等级联动",
    ...{ style: {} },
}));
const __VLS_21 = __VLS_20({
    modelValue: (__VLS_ctx.form.name),
    placeholder: "如 入侵-高等级联动",
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
    label: "类型",
}));
const __VLS_26 = __VLS_25({
    label: "类型",
}, ...__VLS_functionalComponentArgsRest(__VLS_25));
const { default: __VLS_28 } = __VLS_27.slots;
const __VLS_29 = {}.ElInput;
/** @type {[typeof __VLS_components.ElInput, typeof __VLS_components.elInput, ]} */ ;
// @ts-ignore
ElInput;
// @ts-ignore
const __VLS_30 = __VLS_asFunctionalComponent(__VLS_29, new __VLS_29({
    modelValue: (__VLS_ctx.form.eventType),
    placeholder: "告警/烟雾/入侵",
    ...{ style: {} },
}));
const __VLS_31 = __VLS_30({
    modelValue: (__VLS_ctx.form.eventType),
    placeholder: "告警/烟雾/入侵",
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
const __VLS_35 = __VLS_asFunctionalComponent(__VLS_34, new __VLS_34({
    label: "等级",
}));
const __VLS_36 = __VLS_35({
    label: "等级",
}, ...__VLS_functionalComponentArgsRest(__VLS_35));
const { default: __VLS_38 } = __VLS_37.slots;
const __VLS_39 = {}.ElSelect;
/** @type {[typeof __VLS_components.ElSelect, typeof __VLS_components.elSelect, typeof __VLS_components.ElSelect, typeof __VLS_components.elSelect, ]} */ ;
// @ts-ignore
ElSelect;
// @ts-ignore
const __VLS_40 = __VLS_asFunctionalComponent(__VLS_39, new __VLS_39({
    modelValue: (__VLS_ctx.form.level),
    placeholder: "选择",
    ...{ style: {} },
}));
const __VLS_41 = __VLS_40({
    modelValue: (__VLS_ctx.form.level),
    placeholder: "选择",
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_40));
const { default: __VLS_43 } = __VLS_42.slots;
// @ts-ignore
[form,];
const __VLS_44 = {}.ElOption;
/** @type {[typeof __VLS_components.ElOption, typeof __VLS_components.elOption, ]} */ ;
// @ts-ignore
ElOption;
// @ts-ignore
const __VLS_45 = __VLS_asFunctionalComponent(__VLS_44, new __VLS_44({
    label: "低",
    value: "低",
}));
const __VLS_46 = __VLS_45({
    label: "低",
    value: "低",
}, ...__VLS_functionalComponentArgsRest(__VLS_45));
const __VLS_49 = {}.ElOption;
/** @type {[typeof __VLS_components.ElOption, typeof __VLS_components.elOption, ]} */ ;
// @ts-ignore
ElOption;
// @ts-ignore
const __VLS_50 = __VLS_asFunctionalComponent(__VLS_49, new __VLS_49({
    label: "中",
    value: "中",
}));
const __VLS_51 = __VLS_50({
    label: "中",
    value: "中",
}, ...__VLS_functionalComponentArgsRest(__VLS_50));
const __VLS_54 = {}.ElOption;
/** @type {[typeof __VLS_components.ElOption, typeof __VLS_components.elOption, ]} */ ;
// @ts-ignore
ElOption;
// @ts-ignore
const __VLS_55 = __VLS_asFunctionalComponent(__VLS_54, new __VLS_54({
    label: "高",
    value: "高",
}));
const __VLS_56 = __VLS_55({
    label: "高",
    value: "高",
}, ...__VLS_functionalComponentArgsRest(__VLS_55));
var __VLS_42;
var __VLS_37;
const __VLS_59 = {}.ElFormItem;
/** @type {[typeof __VLS_components.ElFormItem, typeof __VLS_components.elFormItem, typeof __VLS_components.ElFormItem, typeof __VLS_components.elFormItem, ]} */ ;
// @ts-ignore
ElFormItem;
// @ts-ignore
const __VLS_60 = __VLS_asFunctionalComponent(__VLS_59, new __VLS_59({
    label: "动作",
}));
const __VLS_61 = __VLS_60({
    label: "动作",
}, ...__VLS_functionalComponentArgsRest(__VLS_60));
const { default: __VLS_63 } = __VLS_62.slots;
const __VLS_64 = {}.ElSelect;
/** @type {[typeof __VLS_components.ElSelect, typeof __VLS_components.elSelect, typeof __VLS_components.ElSelect, typeof __VLS_components.elSelect, ]} */ ;
// @ts-ignore
ElSelect;
// @ts-ignore
const __VLS_65 = __VLS_asFunctionalComponent(__VLS_64, new __VLS_64({
    modelValue: (__VLS_ctx.form.actions),
    multiple: true,
    placeholder: "选择",
    ...{ style: {} },
}));
const __VLS_66 = __VLS_65({
    modelValue: (__VLS_ctx.form.actions),
    multiple: true,
    placeholder: "选择",
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_65));
const { default: __VLS_68 } = __VLS_67.slots;
// @ts-ignore
[form,];
const __VLS_69 = {}.ElOption;
/** @type {[typeof __VLS_components.ElOption, typeof __VLS_components.elOption, ]} */ ;
// @ts-ignore
ElOption;
// @ts-ignore
const __VLS_70 = __VLS_asFunctionalComponent(__VLS_69, new __VLS_69({
    label: "弹窗",
    value: "弹窗",
}));
const __VLS_71 = __VLS_70({
    label: "弹窗",
    value: "弹窗",
}, ...__VLS_functionalComponentArgsRest(__VLS_70));
const __VLS_74 = {}.ElOption;
/** @type {[typeof __VLS_components.ElOption, typeof __VLS_components.elOption, ]} */ ;
// @ts-ignore
ElOption;
// @ts-ignore
const __VLS_75 = __VLS_asFunctionalComponent(__VLS_74, new __VLS_74({
    label: "短信",
    value: "短信",
}));
const __VLS_76 = __VLS_75({
    label: "短信",
    value: "短信",
}, ...__VLS_functionalComponentArgsRest(__VLS_75));
const __VLS_79 = {}.ElOption;
/** @type {[typeof __VLS_components.ElOption, typeof __VLS_components.elOption, ]} */ ;
// @ts-ignore
ElOption;
// @ts-ignore
const __VLS_80 = __VLS_asFunctionalComponent(__VLS_79, new __VLS_79({
    label: "联动门禁",
    value: "联动门禁",
}));
const __VLS_81 = __VLS_80({
    label: "联动门禁",
    value: "联动门禁",
}, ...__VLS_functionalComponentArgsRest(__VLS_80));
var __VLS_67;
var __VLS_62;
const __VLS_84 = {}.ElFormItem;
/** @type {[typeof __VLS_components.ElFormItem, typeof __VLS_components.elFormItem, typeof __VLS_components.ElFormItem, typeof __VLS_components.elFormItem, ]} */ ;
// @ts-ignore
ElFormItem;
// @ts-ignore
const __VLS_85 = __VLS_asFunctionalComponent(__VLS_84, new __VLS_84({}));
const __VLS_86 = __VLS_85({}, ...__VLS_functionalComponentArgsRest(__VLS_85));
const { default: __VLS_88 } = __VLS_87.slots;
const __VLS_89 = {}.ElButton;
/** @type {[typeof __VLS_components.ElButton, typeof __VLS_components.elButton, typeof __VLS_components.ElButton, typeof __VLS_components.elButton, ]} */ ;
// @ts-ignore
ElButton;
// @ts-ignore
const __VLS_90 = __VLS_asFunctionalComponent(__VLS_89, new __VLS_89({
    ...{ 'onClick': {} },
    type: "primary",
    disabled: (!__VLS_ctx.form.name),
}));
const __VLS_91 = __VLS_90({
    ...{ 'onClick': {} },
    type: "primary",
    disabled: (!__VLS_ctx.form.name),
}, ...__VLS_functionalComponentArgsRest(__VLS_90));
let __VLS_93;
let __VLS_94;
const __VLS_95 = ({ click: {} },
    { onClick: (__VLS_ctx.add) });
const { default: __VLS_96 } = __VLS_92.slots;
// @ts-ignore
[form, add,];
var __VLS_92;
var __VLS_87;
var __VLS_9;
const __VLS_97 = {}.ElTable;
/** @type {[typeof __VLS_components.ElTable, typeof __VLS_components.elTable, typeof __VLS_components.ElTable, typeof __VLS_components.elTable, ]} */ ;
// @ts-ignore
ElTable;
// @ts-ignore
const __VLS_98 = __VLS_asFunctionalComponent(__VLS_97, new __VLS_97({
    data: (__VLS_ctx.strategyStore.strategies),
    size: "small",
    ...{ style: {} },
}));
const __VLS_99 = __VLS_98({
    data: (__VLS_ctx.strategyStore.strategies),
    size: "small",
    ...{ style: {} },
}, ...__VLS_functionalComponentArgsRest(__VLS_98));
const { default: __VLS_101 } = __VLS_100.slots;
// @ts-ignore
[strategyStore,];
const __VLS_102 = {}.ElTableColumn;
/** @type {[typeof __VLS_components.ElTableColumn, typeof __VLS_components.elTableColumn, ]} */ ;
// @ts-ignore
ElTableColumn;
// @ts-ignore
const __VLS_103 = __VLS_asFunctionalComponent(__VLS_102, new __VLS_102({
    prop: "name",
    label: "名称",
    width: "180",
}));
const __VLS_104 = __VLS_103({
    prop: "name",
    label: "名称",
    width: "180",
}, ...__VLS_functionalComponentArgsRest(__VLS_103));
const __VLS_107 = {}.ElTableColumn;
/** @type {[typeof __VLS_components.ElTableColumn, typeof __VLS_components.elTableColumn, typeof __VLS_components.ElTableColumn, typeof __VLS_components.elTableColumn, ]} */ ;
// @ts-ignore
ElTableColumn;
// @ts-ignore
const __VLS_108 = __VLS_asFunctionalComponent(__VLS_107, new __VLS_107({
    label: "条件",
}));
const __VLS_109 = __VLS_108({
    label: "条件",
}, ...__VLS_functionalComponentArgsRest(__VLS_108));
const { default: __VLS_111 } = __VLS_110.slots;
{
    const { default: __VLS_112 } = __VLS_110.slots;
    const [{ row }] = __VLS_getSlotParameters(__VLS_112);
    (row.condition.eventType || '任意');
    (row.condition.level || '任意');
}
var __VLS_110;
const __VLS_113 = {}.ElTableColumn;
/** @type {[typeof __VLS_components.ElTableColumn, typeof __VLS_components.elTableColumn, typeof __VLS_components.ElTableColumn, typeof __VLS_components.elTableColumn, ]} */ ;
// @ts-ignore
ElTableColumn;
// @ts-ignore
const __VLS_114 = __VLS_asFunctionalComponent(__VLS_113, new __VLS_113({
    prop: "actions",
    label: "动作",
    width: "220",
}));
const __VLS_115 = __VLS_114({
    prop: "actions",
    label: "动作",
    width: "220",
}, ...__VLS_functionalComponentArgsRest(__VLS_114));
const { default: __VLS_117 } = __VLS_116.slots;
{
    const { default: __VLS_118 } = __VLS_116.slots;
    const [{ row }] = __VLS_getSlotParameters(__VLS_118);
    (row.actions.join('、'));
}
var __VLS_116;
const __VLS_119 = {}.ElTableColumn;
/** @type {[typeof __VLS_components.ElTableColumn, typeof __VLS_components.elTableColumn, typeof __VLS_components.ElTableColumn, typeof __VLS_components.elTableColumn, ]} */ ;
// @ts-ignore
ElTableColumn;
// @ts-ignore
const __VLS_120 = __VLS_asFunctionalComponent(__VLS_119, new __VLS_119({
    label: "操作",
    width: "100",
}));
const __VLS_121 = __VLS_120({
    label: "操作",
    width: "100",
}, ...__VLS_functionalComponentArgsRest(__VLS_120));
const { default: __VLS_123 } = __VLS_122.slots;
{
    const { default: __VLS_124 } = __VLS_122.slots;
    const [{ row }] = __VLS_getSlotParameters(__VLS_124);
    const __VLS_125 = {}.ElPopconfirm;
    /** @type {[typeof __VLS_components.ElPopconfirm, typeof __VLS_components.elPopconfirm, typeof __VLS_components.ElPopconfirm, typeof __VLS_components.elPopconfirm, ]} */ ;
    // @ts-ignore
    ElPopconfirm;
    // @ts-ignore
    const __VLS_126 = __VLS_asFunctionalComponent(__VLS_125, new __VLS_125({
        ...{ 'onConfirm': {} },
        title: "删除该策略？",
    }));
    const __VLS_127 = __VLS_126({
        ...{ 'onConfirm': {} },
        title: "删除该策略？",
    }, ...__VLS_functionalComponentArgsRest(__VLS_126));
    let __VLS_129;
    let __VLS_130;
    const __VLS_131 = ({ confirm: {} },
        { onConfirm: (...[$event]) => {
                __VLS_ctx.remove(row.id);
                // @ts-ignore
                [remove,];
            } });
    const { default: __VLS_132 } = __VLS_128.slots;
    {
        const { reference: __VLS_133 } = __VLS_128.slots;
        const __VLS_134 = {}.ElButton;
        /** @type {[typeof __VLS_components.ElButton, typeof __VLS_components.elButton, typeof __VLS_components.ElButton, typeof __VLS_components.elButton, ]} */ ;
        // @ts-ignore
        ElButton;
        // @ts-ignore
        const __VLS_135 = __VLS_asFunctionalComponent(__VLS_134, new __VLS_134({
            size: "small",
            type: "danger",
        }));
        const __VLS_136 = __VLS_135({
            size: "small",
            type: "danger",
        }, ...__VLS_functionalComponentArgsRest(__VLS_135));
        const { default: __VLS_138 } = __VLS_137.slots;
        var __VLS_137;
    }
    var __VLS_128;
}
var __VLS_122;
var __VLS_100;
var __VLS_3;
var __VLS_dollars;
const __VLS_self = (await import('vue')).defineComponent({
    setup() {
        return {
            strategyStore: strategyStore,
            form: form,
            add: add,
            remove: remove,
        };
    },
});
export default (await import('vue')).defineComponent({
    setup() {
    },
});
; /* PartiallyEnd: #4569/main.vue */
