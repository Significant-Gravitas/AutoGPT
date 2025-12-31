import {
  descriptionId,
  FieldProps,
  FormContextType,
  getTemplate,
  getUiOptions,
  RJSFSchema,
  StrictRJSFSchema,
  titleId,
} from "@rjsf/utils";
import { shouldShowTypeSelector } from "../helpers";
import { useIsArrayItem } from "../../array/context/array-item-context";
import { getHandleId } from "../../common/field-templates/helpers";

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
>(
  props: customFieldProps<T, S, F>,
) => {
  const { uiSchema, schema, required, name, registry, fieldPathId, selector } =
    props;

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

  const isArrayItem = useIsArrayItem();
  const shouldShowSelector = shouldShowTypeSelector(schema) && !isArrayItem;

  return (
    <div className="flex items-center justify-between gap-2">
      <div className="flex items-center">
        <TitleFieldTemplate
          id={title_id}
          title={schema.title || name || ""}
          required={required}
          schema={schema}
          registry={registry}
          uiSchema={uiSchema}
        />
        <DescriptionFieldTemplate
          id={description_id}
          description={schema.description || ""}
          schema={schema}
          registry={registry}
        />
      </div>
      {shouldShowSelector && selector}
    </div>
  );
};
