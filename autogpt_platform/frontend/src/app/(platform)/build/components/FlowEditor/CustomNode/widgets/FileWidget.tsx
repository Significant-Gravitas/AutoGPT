import { WidgetProps } from "@rjsf/utils";
import { InputRenderer, InputType } from "../InputRenderer";

export const FileWidget = (props: WidgetProps) => {
  const { onChange, multiple = false, disabled, readonly } = props;

  // TODO: Need a lot of work here
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
    <InputRenderer
      type={InputType.FILE}
      id={props.id}
      value={props.value}
      onChange={handleChange}
      disabled={props.disabled}
      readonly={props.readonly}
      placeholder={props.placeholder || ""}
      required={props.required}
      autofocus={props.autofocus}
      multiple={multiple}
    />
  );
};
