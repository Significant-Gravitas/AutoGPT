import {
  ADDITIONAL_PROPERTY_FLAG,
  descriptionId,
  getUiOptions,
  TitleFieldProps,
} from "@rjsf/utils";

import { Text } from "@/components/atoms/Text/Text";
import { getTypeDisplayInfo } from "@/app/(platform)/build/components/FlowEditor/nodes/helpers";
import { isAnyOfSchema } from "../../utils/schema-utils";
import { useIsArrayItem } from "../array/context/array-item-context";
import { cn } from "@/lib/utils";
import { cleanUpHandleId } from "../../helpers";

export default function TitleField(props: TitleFieldProps) {
  const { id, title, required, schema, registry, uiSchema } = props;
  const { displayType, colorClass } = getTypeDisplayInfo(schema);
  const { nodeId } = registry.formContext;

  const isAnyOf = isAnyOfSchema(schema);
  const description_id = descriptionId(id);

  const isArrayItem = useIsArrayItem();
  const additional = ADDITIONAL_PROPERTY_FLAG in schema;
  const smallText = isArrayItem || additional;

  const uiOptions = getUiOptions(uiSchema);
  const handleId = cleanUpHandleId(uiOptions.handleId);
  return (
    <div className="flex items-center gap-1">
      {/* Add node handle here */}
      <Text
        variant={isArrayItem ? "small" : "body"}
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
