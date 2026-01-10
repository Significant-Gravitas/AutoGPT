import { GoogleDrivePickerInput } from "@/components/contextual/GoogleDrivePicker/GoogleDrivePickerInput";
import { GoogleDrivePickerConfig } from "@/lib/autogpt-server-api";
import { FieldProps, getUiOptions } from "@rjsf/utils";

export const GoogleDrivePickerField = (props: FieldProps) => {
  const { schema, uiSchema, onChange, fieldPathId, formData } = props;
  const uiOptions = getUiOptions(uiSchema);
  const config: GoogleDrivePickerConfig = schema.google_drive_picker_config;

  return (
    <div>
      <GoogleDrivePickerInput
        config={config}
        value={formData}
        onChange={(value) => onChange(value, fieldPathId.path)}
        className={uiOptions.className}
        showRemoveButton={true}
      />
    </div>
  );
};
