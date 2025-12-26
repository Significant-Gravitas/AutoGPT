import {
  descriptionId,
  FieldProps,
  FormContextType,
  getTemplate,
  getUiOptions,
  getWidget,
  RJSFSchema,
  StrictRJSFSchema,
  titleId,
} from "@rjsf/utils";

interface customFieldProps<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
> extends FieldProps<T, S, F> {
  selector: JSX.Element;
}

export const AnyOfFieldTitle = <
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>({
  uiSchema,
  schema,
  required,
  name,
  registry,
  fieldPathId,
  selector,
}: customFieldProps<T, S, F>) => {
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

  const title_id = titleId(fieldPathId ?? "");
  const description_id = descriptionId(fieldPathId ?? "");
  return (
    <div className="flex items-center justify-between gap-2">
      <div className="flex items-center">
        <TitleFieldTemplate
          id={title_id}
          title={schema.title || name || ""}
          required={required}
          schema={schema}
          registry={registry}
        />
        <DescriptionFieldTemplate
          id={description_id}
          description={schema.description || ""}
          schema={schema}
          registry={registry}
        />
      </div>

      {selector}
    </div>
  );
};
