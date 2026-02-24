import {
  ADDITIONAL_PROPERTY_FLAG,
  buttonId,
  canExpand,
  descriptionId,
  getTemplate,
  getUiOptions,
  ObjectFieldTemplateProps,
  titleId,
} from "@rjsf/utils";
import { cleanUpHandleId, getHandleId, updateUiOption } from "../../helpers";
import React from "react";
import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";

export default function ObjectFieldTemplate(props: ObjectFieldTemplateProps) {
  const {
    description,
    title,
    properties,
    required,
    uiSchema,
    fieldPathId,
    schema,
    formData,
    optionalDataControl,
    onAddProperty,
    disabled,
    readonly,
    registry,
  } = props;
  const uiOptions = getUiOptions(uiSchema);

  const { isInputConnected } = useEdgeStore();

  const TitleFieldTemplate = getTemplate(
    "TitleFieldTemplate",
    registry,
    uiOptions,
  );

  const DescriptionFieldTemplate = getTemplate(
    "DescriptionFieldTemplate",
    registry,
    uiOptions,
  );
  const showOptionalDataControlInTitle = !readonly && !disabled;

  const {
    ButtonTemplates: { AddButton },
  } = registry.templates;

  const additional = ADDITIONAL_PROPERTY_FLAG in schema;

  const handleId = getHandleId({
    uiOptions,
    id: fieldPathId.$id,
    schema,
  });

  const updatedUiSchema = updateUiOption(uiSchema, {
    handleId: handleId,
  });

  const shouldShowChildren = !isInputConnected(
    registry.formContext.nodeId,
    cleanUpHandleId(handleId),
  );

  return (
    <>
      <div className="flex items-center gap-2">
        {title && !additional && (
          <TitleFieldTemplate
            id={titleId(fieldPathId)}
            title={title}
            required={required}
            schema={schema}
            uiSchema={updatedUiSchema}
            registry={registry}
            optionalDataControl={true ? optionalDataControl : undefined}
          />
        )}
        {description && (
          <DescriptionFieldTemplate
            id={descriptionId(fieldPathId)}
            description={description}
            schema={schema}
            uiSchema={updatedUiSchema}
            registry={registry}
          />
        )}
      </div>

      {shouldShowChildren && (
        <div className="flex flex-col">
          {!showOptionalDataControlInTitle ? optionalDataControl : undefined}

          {/* I have cloned it - so i could pass updated uiSchema to the nested children */}
          {properties.map((element: any, index: number) => {
            const clonedContent = React.cloneElement(element.content, {
              ...element.content.props,
              uiSchema: updateUiOption(element.content.props.uiSchema, {
                handleId: handleId,
              }),
            });

            return (
              <div
                key={index}
                className={`${element.hidden ? "hidden" : ""} flex`}
              >
                <div className="w-full">{clonedContent}</div>
              </div>
            );
          })}
          {canExpand(schema, uiSchema, formData) ? (
            <div className="mt-2 flex justify-end">
              <AddButton
                id={buttonId(fieldPathId, "add")}
                onClick={onAddProperty}
                disabled={disabled || readonly}
                className="rjsf-object-property-expand"
                uiSchema={updatedUiSchema}
                registry={registry}
              />
            </div>
          ) : null}
        </div>
      )}
    </>
  );
}
