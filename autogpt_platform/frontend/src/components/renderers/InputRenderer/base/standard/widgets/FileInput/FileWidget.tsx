import { WidgetProps } from "@rjsf/utils";
import { FileInput } from "@/components/atoms/FileInput/FileInput";
import { useWorkspaceUpload } from "./useWorkspaceUpload";

export function FileWidget(props: WidgetProps) {
  const { onChange, disabled, readonly, value, schema, formContext } = props;
  const { size } = formContext || {};
  const displayName = schema?.title || "File";
  const { handleUploadFile, handleDeleteFile } = useWorkspaceUpload();

  function handleChange(fileURI: string) {
    onChange(fileURI);
  }

  return (
    <FileInput
      variant={size === "large" ? "default" : "compact"}
      mode="upload"
      value={value}
      placeholder={displayName}
      onChange={handleChange}
      onDeleteFile={handleDeleteFile}
      onUploadFile={handleUploadFile}
      showStorageNote={false}
      className={
        disabled || readonly ? "pointer-events-none opacity-50" : undefined
      }
    />
  );
}
