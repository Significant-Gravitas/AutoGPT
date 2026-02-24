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
    <div>
      <div className="mb-2 flex flex-row flex-wrap items-center">
        <div className="shrink grow">
          <div className="shrink grow">{children}</div>
        </div>
        <div className="flex items-end justify-end">
          {hasToolbar && (
            <div className="-mt-4 mb-2 flex gap-2">
              <ArrayFieldItemButtonsTemplate {...buttonsProps} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
