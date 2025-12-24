import {
  FormContextType,
  RJSFSchema,
  StrictRJSFSchema,
  TitleFieldProps,
} from "@rjsf/utils";

import { Text } from "@/components/atoms/Text/Text";
import { getTypeDisplayInfo } from "@/app/(platform)/build/components/FlowEditor/nodes/helpers";

export default function TitleField<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>({ id, title, required, schema }: TitleFieldProps<T, S, F>) {
  const { displayType, colorClass } = getTypeDisplayInfo(schema);
  return (
    <div className="flex items-center gap-1">
      <Text variant="body" id={id} className="line-clamp-1">
        {title}
      </Text>
      <Text variant="small" className={colorClass}>
        ({displayType})
      </Text>
      <Text variant="small" className={"text-red-500"}>
        {required ? "*" : null}
      </Text>
    </div>
  );
}
