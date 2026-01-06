import {
  ADDITIONAL_PROPERTY_FLAG,
  descriptionId,
  getUiOptions,
  TitleFieldProps,
} from "@rjsf/utils";

import { Text } from "@/components/atoms/Text/Text";
import { getTypeDisplayInfo } from "@/app/(platform)/build/components/FlowEditor/nodes/helpers";
import { isAnyOfSchema } from "../../utils/schema-utils";
import { cn } from "@/lib/utils";
import { isArrayItem } from "../../helpers";
import { InputNodeHandle } from "@/app/(platform)/build/components/FlowEditor/handlers/NodeHandle";

export default function TitleField(props: TitleFieldProps) {
  const { id, title, required, schema, registry, uiSchema } = props;
  const { nodeId, showHandles } = registry.formContext;
  const uiOptions = getUiOptions(uiSchema);

  const isAnyOf = isAnyOfSchema(schema);
  const { displayType, colorClass } = getTypeDisplayInfo(schema);
  const description_id = descriptionId(id);

  const additional = ADDITIONAL_PROPERTY_FLAG in schema;
  const isArrayItemFlag = isArrayItem({ uiOptions });
  const smallText = isArrayItemFlag || additional;

  const showHandle = uiOptions.showHandles ?? showHandles;
  return (
    <div className="flex items-center">
      {showHandle !== false && (
        <InputNodeHandle handleId={uiOptions.handleId} nodeId={nodeId} />
      )}
      <Text
        variant={isArrayItemFlag ? "small" : "body"}
        id={id}
        className={cn("line-clamp-1", smallText && "text-sm text-zinc-700")}
      >
        {title}
      </Text>
      <Text variant="small" className={"mr-1 text-red-500"}>
        {required ? "*" : null}
      </Text>
      {!isAnyOf && (
        <Text
          variant="small"
          className={cn("ml-2", colorClass)}
          id={description_id}
        >
          ({displayType})
        </Text>
      )}
    </div>
  );
}
