import {
  ADDITIONAL_PROPERTY_FLAG,
  buttonId,
  FormContextType,
  getTemplate,
  getUiOptions,
  RJSFSchema,
  StrictRJSFSchema,
  titleId,
  TranslatableString,
  WrapIfAdditionalTemplateProps,
} from "@rjsf/utils";

import { Separator } from "@/components/__legacy__/ui/separator";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";

/** The `WrapIfAdditional` component is used by the `FieldTemplate` to rename, or remove properties that are
 * part of an `additionalProperties` part of a schema.
 *
 * @param props - The `WrapIfAdditionalProps` for this component
 */
export default function WrapIfAdditionalTemplate<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>({
  classNames,
  style,
  children,
  disabled,
  id,
  label,
  displayLabel,
  onRemoveProperty,
  onKeyRenameBlur,
  rawDescription,
  readonly,
  required,
  schema,
  uiSchema,
  registry,
}: WrapIfAdditionalTemplateProps<T, S, F>) {
  const { templates, translateString } = registry;
  // Button templates are not overridden in the uiSchema
  const { RemoveButton } = templates.ButtonTemplates;
  const additional = ADDITIONAL_PROPERTY_FLAG in schema;

  const TitleFieldTemplate = getTemplate<"TitleFieldTemplate", T, S, F>(
    "TitleFieldTemplate",
    registry,
    getUiOptions(uiSchema),
  );

  if (!additional) {
    return (
      <div className={classNames} style={style}>
        {children}
      </div>
    );
  }

  const marginDesc = rawDescription ? -28 : 0;
  const margin = displayLabel ? 22 + marginDesc : 0;
  const keyId = `${id}-key`;

  return (
    <>
      <div className={`mb-4 flex flex-col gap-1`} style={style}>
        <TitleFieldTemplate
          id={titleId(id)}
          title={label}
          required={required}
          schema={schema}
          registry={registry}
        />
        <div className="flex flex-1 items-center gap-2">
          <Input
            label={""}
            hideLabel={true}
            required={required}
            defaultValue={label}
            disabled={disabled || readonly}
            id={keyId}
            wrapperClassName="mb-2 w-30"
            name={keyId}
            onBlur={!readonly ? onKeyRenameBlur : undefined}
            type="text"
            size="small"
          />
          {children}
        </div>

        <RemoveButton
          id={buttonId(id, "remove")}
          disabled={disabled || readonly}
          onClick={onRemoveProperty}
          uiSchema={uiSchema}
          registry={registry}
        />
      </div>
    </>
  );
}
