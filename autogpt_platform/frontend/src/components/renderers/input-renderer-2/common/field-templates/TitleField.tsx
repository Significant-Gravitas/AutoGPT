import {
  ADDITIONAL_PROPERTY_FLAG,
  descriptionId,
  RJSFSchema,
  StrictRJSFSchema,
  TitleFieldProps,
  titleId,
} from "@rjsf/utils";

import { Text } from "@/components/atoms/Text/Text";
import { getTypeDisplayInfo } from "@/app/(platform)/build/components/FlowEditor/nodes/helpers";
import { isAnyOfSchema } from "../../utils/schema-utils";
import { InputNodeHandle } from "@/app/(platform)/build/components/FlowEditor/handlers/NodeHandle";
import { ExtendedFormContextType } from "../../types";
import { useIsArrayItem } from "../../array/context/array-item-context";
import { cn } from "@/lib/utils";
import { getHandleId } from "./helpers";

export default function TitleField<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends ExtendedFormContextType = ExtendedFormContextType,
>(props: TitleFieldProps<T, S, F>) {
  const { id, title, required, schema, registry } = props;
  const { displayType, colorClass } = getTypeDisplayInfo(schema);
  const { nodeId } = registry.formContext;

  const isAnyOf = isAnyOfSchema(schema);
  const isArrayItem = useIsArrayItem();
  const description_id = descriptionId(id);

  const additional = ADDITIONAL_PROPERTY_FLAG in schema;

  const handleId = getHandleId(registry.rootSchema, id);
  console.log("handleId", handleId);
  return (
    <div className="flex items-center gap-1">
      <InputNodeHandle
        titleId={id}
        additional={additional}
        nodeId={nodeId ?? ""}
      />
      <Text
        variant={isArrayItem ? "small" : "body"}
        id={id}
        className={cn("line-clamp-1", isArrayItem && "text-zinc-700")}
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
