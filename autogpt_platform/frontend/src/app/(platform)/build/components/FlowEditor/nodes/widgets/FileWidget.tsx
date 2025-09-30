import { WidgetProps } from "@rjsf/utils";
import { Input } from "@/components/__legacy__/ui/input";

export const FileWidget = (props: WidgetProps) => {
  const { onChange, multiple = false, disabled, readonly, id } = props;

  // TODO: It's temporary solution for file input, will complete it follow up prs
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
      id={id}
      type="file"
      multiple={multiple}
      disabled={disabled || readonly}
      onChange={handleChange}
      className="rounded-full"
    />
  );
};
