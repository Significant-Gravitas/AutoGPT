import {
  ArrayFieldItemTemplateProps,
  getTemplate,
  getUiOptions,
} from "@rjsf/utils";

export default function ArrayFieldItemTemplate(
  props: ArrayFieldItemTemplateProps,
) {
  const { children, buttonsProps, hasToolbar, uiSchema, registry } = props;
  const uiOptions = getUiOptions(uiSchema);
  const ArrayFieldItemButtonsTemplate = getTemplate(
    "ArrayFieldItemButtonsTemplate",
    registry,
    uiOptions,
  );

  return (
    <div className="mb-4 flex flex-col">
      <div className="w-full">{children}</div>
      {hasToolbar && (
        <div className="-mt-2 flex justify-start gap-2">
          <ArrayFieldItemButtonsTemplate {...buttonsProps} />
        </div>
      )}
    </div>
  );
}
