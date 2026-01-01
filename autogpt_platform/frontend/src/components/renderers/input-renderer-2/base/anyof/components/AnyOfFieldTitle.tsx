import {
  descriptionId,
  FieldProps,
  getTemplate,
  getUiOptions,
  titleId,
} from "@rjsf/utils";
import { shouldShowTypeSelector } from "../helpers";
import { useIsArrayItem } from "../../array/context/array-item-context";

interface customFieldProps extends FieldProps {
  selector: JSX.Element;
}

export const AnyOfFieldTitle = (props: customFieldProps) => {
  const { uiSchema, schema, required, name, registry, fieldPathId, selector } =
    props;

  const uiOptions = getUiOptions(uiSchema);
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
