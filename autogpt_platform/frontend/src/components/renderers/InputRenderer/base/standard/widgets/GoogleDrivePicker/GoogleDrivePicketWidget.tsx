import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { GoogleDrivePickerInput } from "@/components/contextual/GoogleDrivePicker/GoogleDrivePickerInput";
import { getFieldErrorKey } from "@/components/renderers/InputRenderer/utils/helpers";
import type { GoogleDrivePickerConfig } from "@/lib/autogpt-server-api/types";
import { cn } from "@/lib/utils";
import { WidgetProps } from "@rjsf/utils";

function hasGoogleDrivePickerConfig(
  schema: unknown,
): schema is { google_drive_picker_config?: GoogleDrivePickerConfig } {
  return (
    typeof schema === "object" &&
    schema !== null &&
    "google_drive_picker_config" in schema
  );
}

export function GoogleDrivePickerWidget(props: WidgetProps) {
  const { onChange, disabled, readonly, value, schema, id, formContext } =
    props;
  const { nodeId } = formContext || {};

  const nodeErrors = useNodeStore((state) => {
    const node = state.nodes.find((n) => n.id === nodeId);
    return node?.data?.errors;
  });

  const fieldErrorKey = getFieldErrorKey(id ?? "");
  const fieldError =
    nodeErrors?.[fieldErrorKey] ||
    nodeErrors?.[fieldErrorKey.replace(/_/g, ".")] ||
    nodeErrors?.[fieldErrorKey.replace(/\./g, "_")] ||
    undefined;

  const config: GoogleDrivePickerConfig = hasGoogleDrivePickerConfig(schema)
    ? schema.google_drive_picker_config || {}
    : {};

  function handleChange(newValue: unknown) {
    onChange(newValue);
  }

  return (
    <GoogleDrivePickerInput
      config={config}
      value={value}
      onChange={handleChange}
      error={fieldError}
      className={cn(
        disabled || readonly ? "pointer-events-none opacity-50" : undefined,
      )}
      showRemoveButton={true}
    />
  );
}
