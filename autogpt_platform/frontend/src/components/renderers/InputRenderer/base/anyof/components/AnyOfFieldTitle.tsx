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
import { isOptionalType } from "../../../utils/schema-utils";
import { getTypeDisplayInfo } from "@/app/(platform)/build/components/FlowEditor/nodes/helpers";
import { cn } from "@/lib/utils";

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

  const { isOptional, type } = isOptionalType(schema); // If we have something like int | null = we will treat it as optional int
  const { displayType, colorClass } = getTypeDisplayInfo(type);

  const shouldShowSelector =
    shouldShowTypeSelector(schema) && !isArrayItem && !isHandleConnected;
  const shoudlShowType = isHandleConnected || (isOptional && type);

  return (
    <div className="flex items-center gap-2">
      <TitleFieldTemplate
        id={title_id}
        title={schema.title || name || ""}
        required={required}
        schema={schema}
        registry={registry}
        uiSchema={uiSchema}
      />
      {shoudlShowType && (
        <Text variant="small" className={cn("text-zinc-700", colorClass)}>
          {isOptional ? `(${displayType})` : "(any)"}
        </Text>
      )}
      {shouldShowSelector && selector}
      <DescriptionFieldTemplate
        id={description_id}
        description={schema.description || ""}
        schema={schema}
        registry={registry}
      />
    </div>
  );
};
