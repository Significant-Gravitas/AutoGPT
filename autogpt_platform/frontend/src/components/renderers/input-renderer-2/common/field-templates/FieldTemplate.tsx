import {
  FieldTemplateProps,
  FormContextType,
  getTemplate,
  getUiOptions,
  RJSFSchema,
  StrictRJSFSchema,
} from "@rjsf/utils";
import { isAnyOfSchema } from "../../utils/schema-utils";

/** The `FieldTemplate` component is the template used by `SchemaField` to render any field. It renders the field
 * content, (label, description, children, errors and help) inside a `WrapIfAdditional` component.
 *
 * @param props - The `FieldTemplateProps` for this component
 */
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
  const uiOptions = getUiOptions(uiSchema);
  if (hidden) {
    return <div className="hidden">{children}</div>;
  }
  const TitleFieldTemplate = getTemplate<"TitleFieldTemplate", T, S, F>(
    "TitleFieldTemplate",
    registry,
    uiOptions,
  );

  const isAnyOf = isAnyOfSchema(schema);
  return (
    <div className="mb-2 flex flex-col gap-2">
      {!isAnyOf && (
        <div className="flex items-center gap-2">
          {displayLabel && (
            <TitleFieldTemplate
              id={`${id}_title`}
              title={label}
              required={required}
              schema={schema}
              registry={registry}
            />
          )}
          {displayLabel && rawDescription && <span>{description}</span>}
        </div>
      )}

      {children}
      {errors}
      {help}
    </div>
  );
}
