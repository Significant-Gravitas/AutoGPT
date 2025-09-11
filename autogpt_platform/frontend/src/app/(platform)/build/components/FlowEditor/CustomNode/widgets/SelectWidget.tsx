import { WidgetProps } from "@rjsf/utils";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export const SelectWidget = (props: WidgetProps) => {
  const { options, value, onChange, disabled, readonly, multiple } = props;
  const enumOptions = options.enumOptions || [];

  if (multiple) {
    return (
      <select
        value={value || ""}
        onChange={(event) => {
          const val = event.target.value;
          onChange(val === "" ? undefined : val);
        }}
        disabled={disabled || readonly}
        multiple={multiple}
      >
        <option value="">Select options</option>
        {enumOptions.map(({ value, label }) => (
          <option key={String(value)} value={value}>
            {label}
          </option>
        ))}
      </select>
    );
  }

  return (
    <Select
      value={value || ""}
      onValueChange={(val) => onChange(val === "" ? undefined : val)}
      disabled={disabled || readonly}
    >
      <SelectTrigger>
        <SelectValue placeholder="Select an option" />
      </SelectTrigger>
      <SelectContent>
        {enumOptions.map(({ value, label }) => (
          <SelectItem key={String(value)} value={String(value)}>
            {label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
};
