import {
  descriptionId,
  FieldProps,
  getTemplate,
  getUiOptions,
  titleId,
} from "@rjsf/utils";
import { shouldShowTypeSelector } from "../helpers";
import { useIsArrayItem } from "../../array/context/array-item-context";
import { cleanUpHandleId } from "../../../helpers";
import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";
import { Text } from "@/components/atoms/Text/Text";

interface customFieldProps extends FieldProps {
  selector: JSX.Element;
}

export const AnyOfFieldTitle = (props: customFieldProps) => {
  const { uiSchema, schema, required, name, registry, fieldPathId, selector } =
    props;
  const { isInputConnected } = useEdgeStore();
  const { nodeId } = registry.formContext;

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

  const handleId = cleanUpHandleId(uiOptions.handleId);
  const isHandleConnected = isInputConnected(nodeId, handleId);

  const shouldShowSelector =
    shouldShowTypeSelector(schema) && !isArrayItem && !isHandleConnected;

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
        {isHandleConnected && (
          <Text variant="small" className="mr-2 text-zinc-700">
            (any)
          </Text>
        )}
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
