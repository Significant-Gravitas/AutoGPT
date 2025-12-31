import {
  ADDITIONAL_PROPERTY_FLAG,
  FieldTemplateProps,
  FormContextType,
  getTemplate,
  getUiOptions,
  RJSFSchema,
  StrictRJSFSchema,
  titleId,
} from "@rjsf/utils";

import { isAnyOfChild, isAnyOfSchema } from "../../utils/schema-utils";
import {
  ANY_OF_FLAG,
  ARRAY_ITEM_FLAG,
  getHandleId,
  updateUiOption,
} from "../../helpers";

import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { useIsArrayItem } from "../../array/context/array-item-context";

export default function FieldTemplate<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>(props: FieldTemplateProps<T, S, F>) {
  const {
    id,
    children,
    displayLabel,
    errors,
    help,
    description,
    rawDescription,
    label,
    hidden,
    required,
    schema,
    uiSchema,
    registry,
    classNames,
    style,
    disabled,
    onKeyRename,
    onKeyRenameBlur,
    onRemoveProperty,
    readonly,
  } = props;

  const showAdvanced = useNodeStore(
    (state) => state.nodeAdvancedStates[registry.formContext.nodeId ?? ""],
  );
  const isAdvancedField = (schema as any).advanced === true;
  if (!showAdvanced && isAdvancedField) {
    return null;
  }

  if (hidden) {
    return <div className="hidden">{children}</div>;
  }

  const uiOptions = getUiOptions(uiSchema);
  const TitleFieldTemplate = getTemplate<"TitleFieldTemplate", T, S, F>(
    "TitleFieldTemplate",
    registry,
    uiOptions,
  );
  const WrapIfAdditionalTemplate = getTemplate<
    "WrapIfAdditionalTemplate",
    T,
    S,
    F
  >("WrapIfAdditionalTemplate", registry, uiOptions);

  const additional = ADDITIONAL_PROPERTY_FLAG in schema;

  const handleId = getHandleId({
    uiOptions,
    id: id,
    schema: schema,
  });
  const updatedUiSchema = updateUiOption(uiSchema, {
    handleId: handleId,
  });

  const shouldDisplayLabel =
    displayLabel ||
    (schema.type === "boolean" && !isAnyOfChild(uiSchema as any));
  const shouldShowTitleSection = !isAnyOfSchema(schema) && !additional;

  return (
    <WrapIfAdditionalTemplate
      classNames={classNames}
      style={style}
      disabled={disabled}
      id={id}
      label={label}
      displayLabel={displayLabel}
      onKeyRename={onKeyRename}
      onKeyRenameBlur={onKeyRenameBlur}
      onRemoveProperty={onRemoveProperty}
      rawDescription={rawDescription}
      readonly={readonly}
      required={required}
      schema={schema}
      uiSchema={updatedUiSchema}
      registry={registry}
    >
      <div className="mb-4 flex flex-col gap-2">
        {shouldShowTitleSection && (
          <div className="flex items-center gap-2">
            {shouldDisplayLabel && (
              <TitleFieldTemplate
                id={titleId(id)}
                title={label}
                required={required}
                schema={schema}
                uiSchema={updatedUiSchema}
                registry={registry}
              />
            )}
            {shouldDisplayLabel && rawDescription && <span>{description}</span>}
          </div>
        )}

        {children}
        {errors}
        {help}
      </div>
    </WrapIfAdditionalTemplate>
  );
}
