import { Input } from "@/components/ui/input";
import { FieldProps } from "@rjsf/utils";

export const AnyField = (props: FieldProps) => {
  return (
    <Input
      type="text"
      value={props.formData || ""}
      onChange={(e) => props.onChange(e.target.value)}
      placeholder={props.schema.title || "Enter value"}
    />
  );
};
