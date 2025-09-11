import { Input } from "@/components/ui/input";
import { WidgetProps } from "@rjsf/utils";

export const FileWidget = (props: WidgetProps) => {
  const { onChange, multiple = false, disabled, readonly } = props;

  // For standard RJSF, upload value must be a string (usually base64) or array of strings for multiple
  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) {
      onChange(undefined);
      return;
    }

    const file = files[0];
    const reader = new FileReader();
    reader.onload = (e) => {
      onChange(e.target?.result);
    };
    reader.readAsDataURL(file);
  };

  return (
    <Input
      type="file"
      multiple={multiple}
      disabled={disabled || readonly}
      onChange={handleChange}
    />
  );
};
