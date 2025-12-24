import {
  FieldTemplateProps,
  FormContextType,
  getTemplate,
  getUiOptions,
  RJSFSchema,
  StrictRJSFSchema,
} from "@rjsf/utils";

import { cn } from "../lib/utils";

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
  rawErrors = [],
  errors,
  help,
  description,
  rawDescription,
  classNames,
  style,
  disabled,
  label,
  hidden,
  onKeyRename,
  onKeyRenameBlur,
  onRemoveProperty,
  readonly,
  required,
  schema,
  uiSchema,
  registry,
}: FieldTemplateProps<T, S, F>) {
  const uiOptions = getUiOptions(uiSchema);
  const WrapIfAdditionalTemplate = getTemplate<
    "WrapIfAdditionalTemplate",
    T,
    S,
    F
  >("WrapIfAdditionalTemplate", registry, uiOptions);
  if (hidden) {
    return <div className="hidden">{children}</div>;
  }
  const TitleFieldTemplate = getTemplate<"TitleFieldTemplate", T, S, F>(
    "TitleFieldTemplate",
    registry,
    uiOptions,
  );

  const isCheckbox = uiOptions.widget === "checkbox";
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
      uiSchema={uiSchema}
      registry={registry}
    >
      <div className="flex flex-col gap-2">
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
          {displayLabel && rawDescription && !isCheckbox && (
            <span>{description}</span>
          )}
        </div>

        {children}

        {errors}
        {help}
      </div>
    </WrapIfAdditionalTemplate>
  );
}
