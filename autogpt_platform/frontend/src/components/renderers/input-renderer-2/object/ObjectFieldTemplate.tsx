import {
  ADDITIONAL_PROPERTY_FLAG,
  buttonId,
  canExpand,
  descriptionId,
  FormContextType,
  getTemplate,
  getUiOptions,
  ObjectFieldTemplateProps,
  RJSFSchema,
  StrictRJSFSchema,
  titleId,
} from "@rjsf/utils";
import {
  getHandleId,
  isCredentialFieldSchema,
  KEY_PAIR_FLAG,
  OBJECT_FLAG,
  updateUiOption,
} from "../helpers";
import React from "react";
import { CredentialsField } from "../credentials/CredentialField";
import { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api";
import { CredentialsInput } from "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/modals/CredentialsInputs/CredentialsInputs";

/** The `ObjectFieldTemplate` is the template to use to render all the inner properties of an object along with the
 * title and description if available. If the object is expandable, then an `AddButton` is also rendered after all
 * the properties.
 *
 * @param props - The `ObjectFieldTemplateProps` for this component
 */
export default function ObjectFieldTemplate<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>(props: ObjectFieldTemplateProps<T, S, F>) {
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
  const TitleFieldTemplate = getTemplate<"TitleFieldTemplate", T, S, F>(
    "TitleFieldTemplate",
    registry,
    uiOptions,
  );
  const DescriptionFieldTemplate = getTemplate<
    "DescriptionFieldTemplate",
    T,
    S,
    F
  >("DescriptionFieldTemplate", registry, uiOptions);
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
      <div className="flex flex-col">
        {!showOptionalDataControlInTitle ? optionalDataControl : undefined}

        {/* I have cloned it - so i could pass parentHandleId to the nested children */}
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
    </>
  );
}
