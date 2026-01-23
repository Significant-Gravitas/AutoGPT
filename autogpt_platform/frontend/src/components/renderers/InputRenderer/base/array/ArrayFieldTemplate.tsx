import {
  ArrayFieldTemplateProps,
  buttonId,
  getTemplate,
  getUiOptions,
} from "@rjsf/utils";
import { cleanUpHandleId, getHandleId, updateUiOption } from "../../helpers";
import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";

export default function ArrayFieldTemplate(props: ArrayFieldTemplateProps) {
  const {
    canAdd,
    disabled,
    fieldPathId,
    uiSchema,
    items,
    optionalDataControl,
    onAddClick,
    readonly,
    registry,
    required,
    schema,
    title,
  } = props;

  const uiOptions = getUiOptions(uiSchema);
  const { isInputConnected } = useEdgeStore();

  const ArrayFieldDescriptionTemplate = getTemplate(
    "ArrayFieldDescriptionTemplate",
    registry,
    uiOptions,
  );
  const ArrayFieldTitleTemplate = getTemplate(
    "ArrayFieldTitleTemplate",
    registry,
    uiOptions,
  );
  const showOptionalDataControlInTitle = !readonly && !disabled;

  const {
    ButtonTemplates: { AddButton },
  } = registry.templates;

  const { fromAnyOf } = uiOptions;

  const handleId = getHandleId({
    uiOptions,
    id: fieldPathId.$id,
    schema: schema,
  });
  const updatedUiSchema = updateUiOption(uiSchema, {
    handleId: handleId,
  });

  const shouldShowChildren = !isInputConnected(
    registry.formContext.nodeId,
    cleanUpHandleId(handleId),
  );

  return (
    <div>
      <div className="m-0 flex p-0">
        <div className="m-0 w-full space-y-4 p-0">
          {!fromAnyOf && (
            <div className="flex items-center gap-2">
              <ArrayFieldTitleTemplate
                fieldPathId={fieldPathId}
                title={uiOptions.title || title}
                schema={schema}
                uiSchema={updatedUiSchema}
                required={required}
                registry={registry}
                optionalDataControl={
                  showOptionalDataControlInTitle
                    ? optionalDataControl
                    : undefined
                }
              />
              <ArrayFieldDescriptionTemplate
                fieldPathId={fieldPathId}
                description={uiOptions.description || schema.description}
                schema={schema}
                uiSchema={updatedUiSchema}
                registry={registry}
              />
            </div>
          )}
          {shouldShowChildren && (
            <div
              key={`array-item-list-${fieldPathId.$id}`}
              className="m-0 mb-2 w-full p-0"
            >
              {!showOptionalDataControlInTitle
                ? optionalDataControl
                : undefined}
              {items}
              {canAdd && (
                <div className="mt-4 flex justify-end">
                  <AddButton
                    id={buttonId(fieldPathId, "add")}
                    className="rjsf-array-item-add"
                    onClick={onAddClick}
                    disabled={disabled || readonly}
                    uiSchema={updatedUiSchema}
                    registry={registry}
                  />
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
