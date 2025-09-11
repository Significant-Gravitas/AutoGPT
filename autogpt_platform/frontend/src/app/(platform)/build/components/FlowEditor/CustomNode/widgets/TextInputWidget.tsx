import { Input } from "@/components/ui/input";
import { WidgetProps } from "@rjsf/utils";

export const TextInputWidget = (props: WidgetProps) => {
  return (
    <Input
      id={props.id}
      value={props.value || ""}
      onChange={(e) => props.onChange(e.target.value)}
      placeholder={props.placeholder || ""}
      required={props.required}
    />
  );
};
