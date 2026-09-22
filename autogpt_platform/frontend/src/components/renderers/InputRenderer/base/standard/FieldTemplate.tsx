import {
  ADDITIONAL_PROPERTY_FLAG,
  FieldTemplateProps,
  getTemplate,
  getUiOptions,
  titleId,
} from "@rjsf/utils";

import {
  isAnyOfChild,
  isAnyOfSchema,
  isOneOfSchema,
} from "../../utils/schema-utils";
import {
  cleanUpHandleId,
  getHandleId,
  isPartOfAnyOf,
  updateUiOption,
} from "../../helpers";

import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";
import { FieldError } from "./FieldError";
import { BlockUIType } from "@/app/(platform)/build/components/types";
import {
  FieldAccessibilityProvider,
  getFieldDomId,
  useFieldAccessibilityContext,
} from "../../field-accessibility";

export default function FieldTemplate(props: FieldTemplateProps) {
  const {
    id,
    children,
    displayLabel,
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
  const { nodeId, uiType } = registry.formContext;
  const parentAccessibility = useFieldAccessibilityContext(id);

  const { isInputConnected } = useEdgeStore();
  const showAdvanced = useNodeStore(
    (state) => state.nodeAdvancedStates[registry.formContext.nodeId ?? ""],
  );

  if (hidden) {
    return <div className="hidden">{children}</div>;
  }

  if (uiType === BlockUIType.NOTE) {
    return children;
  }

  const uiOptions = getUiOptions(uiSchema);
  const TitleFieldTemplate = getTemplate(
    "TitleFieldTemplate",
    registry,
    uiOptions,
  );
  const WrapIfAdditionalTemplate = getTemplate(
    "WrapIfAdditionalTemplate",
    registry,
    uiOptions,
  );

  const additional = ADDITIONAL_PROPERTY_FLAG in schema;

  const handleId = getHandleId({
    uiOptions,
    id: id,
    schema: schema,
  });
  const updatedUiSchema = updateUiOption(uiSchema, {
    handleId: handleId,
  });
  const isHandleConnected = isInputConnected(nodeId, cleanUpHandleId(handleId));

  const shouldDisplayLabel =
    displayLabel ||
    (schema.type === "boolean" && !isAnyOfChild(uiSchema as any));
  const isUnionSchema = isAnyOfSchema(schema) || isOneOfSchema(schema);
  const shouldShowTitleSection = !isUnionSchema && !additional;

  const shouldShowChildren =
    schema.type === "object" ||
    schema.type === "array" ||
    isUnionSchema ||
    !isHandleConnected;

  const isAdvancedField = (schema as any).advanced === true;
  if (!showAdvanced && isAdvancedField && !isHandleConnected) {
    return null;
  }

  const marginBottom = isPartOfAnyOf({ uiOptions }) || isUnionSchema ? 0 : 16;
  const domId = getFieldDomId(id, registry.formContext);
  const accessibleDescriptionId =
    parentAccessibility?.descriptionId ??
    (rawDescription ? `${domId}-description` : undefined);
  const errorId = parentAccessibility?.errorId ?? `${domId}-error`;

  return (
    <FieldAccessibilityProvider
      value={{
        fieldId: id,
        labelId:
          (shouldShowTitleSection && shouldDisplayLabel) || isUnionSchema
            ? getFieldDomId(titleId(id), registry.formContext)
            : parentAccessibility?.labelId,
        descriptionId: accessibleDescriptionId,
        errorId,
      }}
    >
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
        <div className="flex flex-col gap-2" style={{ marginBottom }}>
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
              {shouldDisplayLabel && rawDescription && (
                <span>{description}</span>
              )}
            </div>
          )}
          {shouldShowChildren && children}

          {rawDescription && !parentAccessibility?.descriptionId && (
            <span id={accessibleDescriptionId} className="sr-only">
              {rawDescription}
            </span>
          )}
          {!parentAccessibility && (
            <FieldError
              nodeId={nodeId}
              fieldId={cleanUpHandleId(id)}
              id={errorId}
            />
          )}
        </div>
      </WrapIfAdditionalTemplate>
    </FieldAccessibilityProvider>
  );
}
