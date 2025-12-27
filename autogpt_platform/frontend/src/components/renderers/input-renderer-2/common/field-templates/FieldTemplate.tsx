import {
  FieldTemplateProps,
  FormContextType,
  getTemplate,
  getUiOptions,
  RJSFSchema,
  StrictRJSFSchema,
  titleId,
} from "@rjsf/utils";
import { isAnyOfChild, isAnyOfSchema } from "../../utils/schema-utils";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";

export default function FieldTemplate<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>({
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
}: FieldTemplateProps<T, S, F>) {
  const showAdvanced = useNodeStore(
    (state) => state.nodeAdvancedStates[registry.formContext.nodeId ?? ""],
  );
  const isAdvancedField = (schema as any).advanced === true; // using any because standard jsonSchema does not have advanced field
  if (!showAdvanced && isAdvancedField) {
    return null;
  }

  if (hidden) {
    return <div className="hidden">{children}</div>;
  }

  const TitleFieldTemplate = getTemplate<"TitleFieldTemplate", T, S, F>(
    "TitleFieldTemplate",
    registry,
    getUiOptions(uiSchema),
  );

  const shouldDisplayLabel =
    displayLabel ||
    (schema.type === "boolean" && !isAnyOfChild(uiSchema as any));

  return (
    <div className="mb-4 flex flex-col gap-2">
      {!isAnyOfSchema(schema) && (
        <div className="flex items-center gap-2">
          {shouldDisplayLabel && (
            <TitleFieldTemplate
              id={titleId(id)}
              title={label}
              required={required}
              schema={schema}
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
  );
}
