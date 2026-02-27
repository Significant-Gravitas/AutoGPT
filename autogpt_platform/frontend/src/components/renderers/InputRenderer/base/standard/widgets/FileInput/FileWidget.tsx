import { WidgetProps } from "@rjsf/utils";
import { FileInput } from "@/components/atoms/FileInput/FileInput";
import { useWorkspaceUpload } from "./useWorkspaceUpload";

export function FileWidget(props: WidgetProps) {
  const { onChange, disabled, readonly, value, schema, formContext } = props;
  const { size } = formContext || {};
  const displayName = schema?.title || "File";
  const { handleUploadFile, uploadProgress } = useWorkspaceUpload();

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
      onUploadFile={handleUploadFile}
      uploadProgress={uploadProgress}
      showStorageNote={false}
      className={
        disabled || readonly ? "pointer-events-none opacity-50" : undefined
      }
    />
  );
}
