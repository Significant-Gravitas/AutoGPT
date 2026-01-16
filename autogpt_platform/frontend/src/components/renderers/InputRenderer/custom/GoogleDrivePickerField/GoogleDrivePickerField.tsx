import { BlockUIType } from "@/app/(platform)/build/components/types";
import { GoogleDrivePickerInput } from "@/components/contextual/GoogleDrivePicker/GoogleDrivePickerInput";
import { GoogleDrivePickerConfig } from "@/lib/autogpt-server-api";
import { FieldProps, getUiOptions } from "@rjsf/utils";

export const GoogleDrivePickerField = (props: FieldProps) => {
  const { schema, uiSchema, onChange, fieldPathId, formData, registry } = props;
  const uiOptions = getUiOptions(uiSchema);
  const config: GoogleDrivePickerConfig = schema.google_drive_picker_config;

  const uiType = registry.formContext?.uiType;

  if (uiType === BlockUIType.INPUT) {
    return (
      <div className="rounded-3xl border border-gray-200 p-2 pl-4 text-xs text-gray-500 hover:cursor-not-allowed">
        Select files when you run the graph
      </div>
    );
  }

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
