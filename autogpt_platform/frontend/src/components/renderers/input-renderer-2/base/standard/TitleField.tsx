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
import { cleanUpHandleId, isArrayItem } from "../../helpers";
import { InputNodeHandle } from "@/app/(platform)/build/components/FlowEditor/handlers/NodeHandle";
import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";

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
    <div className="flex items-center gap-1">
      {showHandle !== false && (
        <InputNodeHandle handleId={uiOptions.handleId} nodeId={nodeId} />
      )}
      <Text
        variant={isArrayItemFlag ? "small" : "body"}
        id={id}
        className={cn("line-clamp-1", smallText && "text-zinc-700")}
      >
        {title}
      </Text>
      {!isAnyOf && (
        <Text variant="small" className={colorClass} id={description_id}>
          ({displayType})
        </Text>
      )}
      <Text variant="small" className={"text-red-500"}>
        {required ? "*" : null}
      </Text>
    </div>
  );
}
