import { useRef, useState } from "react";
import { Button } from "../Button/Button";
import { formatFileSize, getFileLabel } from "./helpers";
import { cn } from "@/lib/utils";
import { parseWorkspaceURI } from "@/lib/workspace-uri";
import { Text } from "../Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { globalRegistry } from "@/components/contextual/OutputRenderers";
import {
  Cancel01Icon,
  Delete02Icon,
  EyeIcon,
  File02Icon,
  Upload01Icon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

function PreviewButton({
  value,
  title,
  contentType,
}: {
  value: string;
  title: string;
  contentType?: string;
}) {
  const metadata = contentType ? { mimeType: contentType } : undefined;
  const renderer = globalRegistry.getRenderer(value, metadata);
  if (!renderer) return null;

  return (
    <Dialog title={title}>
      <Dialog.Trigger>
        <Button
          data-slot="file-input-preview"
          variant="outline"
          size="small"
          className="h-7 w-7 min-w-0 flex-shrink-0 border-border-strong p-0 text-gray-500 hover:text-info-foreground dark:text-gray-400 dark:hover:text-blue-500"
          type="button"
          aria-label="Preview file"
        >
          <Icon data-slot="file-input-preview-icon" icon={EyeIcon} size={14} />
        </Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <div
          data-slot="file-input-preview-body"
          className="overflow-hidden [&>*]:rounded-xlarge"
        >
          {renderer.render(value, metadata)}
        </div>
      </Dialog.Content>
    </Dialog>
  );
}

function getMimeFromDataURI(value: string): string | null {
  const match = value.match(/^data:([^;,]+)/);
  return match?.[1] || null;
}

function isPreviewableFile(
  value: string | undefined,
  contentType: string | undefined,
): boolean {
  if (!value) return false;
  const mimeType =
    contentType ||
    parseWorkspaceURI(value)?.mimeType ||
    (value.startsWith("data:") ? getMimeFromDataURI(value) : null);
  if (!mimeType) return false;
  const normalized = mimeType.toLowerCase();
  return (
    normalized.startsWith("audio/") ||
    normalized.startsWith("video/") ||
    normalized.startsWith("image/")
  );
}

type UploadFileResult = {
  file_name: string;
  size: number;
  content_type: string;
  file_uri: string;
};

type FileInputVariant = "default" | "compact";

interface BaseProps {
  value?: string;
  placeholder?: string;
  onChange: (value: string) => void;
  onDeleteFile?: (fileURI: string) => void;
  className?: string;
  maxFileSize?: number;
  accept?: string | string[];
  variant?: FileInputVariant;
  showStorageNote?: boolean;
}

interface UploadModeProps extends BaseProps {
  mode?: "upload";
  onUploadFile: (file: File) => Promise<UploadFileResult>;
  uploadProgress?: number;
}

interface Base64ModeProps extends BaseProps {
  mode: "base64";
  onUploadFile?: never;
  uploadProgress?: never;
}

type Props = UploadModeProps | Base64ModeProps;

// TODO(tokens): no role holds Tailwind's gray-50/300/500/600/700/900
// (#F9FAFB, #D1D5DB, #6B7280, #4B5563, #374151, #111827) or blue-200
// #BFDBFE, blue-300 #93C5FD, blue-700 #1D4ED8. The dark: classes never apply
// (forcedTheme="light") and are DS6's (SECRT-3116) to delete.
export function FileInput(props: Props) {
  const {
    value,
    onChange,
    onDeleteFile,
    className,
    maxFileSize,
    accept,
    placeholder,
    variant = "default",
    showStorageNote = true,
    mode = "upload",
  } = props;

  const onUploadFile =
    mode === "upload" ? (props as UploadModeProps).onUploadFile : undefined;

  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [fileInfo, setFileInfo] = useState<{
    name: string;
    size: number;
    content_type: string;
  } | null>(null);

  const inputRef = useRef<HTMLInputElement>(null);

  const storageNote = "Files are stored securely in your workspace.";

  function acceptToString(a?: string | string[]) {
    if (!a) return "*/*";
    return Array.isArray(a) ? a.join(",") : a;
  }

  function isAcceptedType(file: File, a?: string | string[]) {
    if (!a) return true;
    const list = Array.isArray(a) ? a : a.split(",").map((s) => s.trim());
    const fileType = file.type;
    const fileExt = file.name.includes(".")
      ? `.${file.name.split(".").pop()}`.toLowerCase()
      : "";

    for (const entry of list) {
      if (!entry) continue;
      const e = entry.toLowerCase();
      if (e.includes("/")) {
        const [main, sub] = e.split("/");
        const [fMain, fSub] = fileType.toLowerCase().split("/");
        if (!fMain || !fSub) continue;
        if (sub === "*") {
          if (main === fMain) return true;
        } else {
          if (e === fileType.toLowerCase()) return true;
        }
      } else if (e.startsWith(".")) {
        if (fileExt === e) return true;
      }
    }
    return false;
  }

  function getFileLabelFromValue(val: unknown): string {
    // Handle object format from external API: { name, type, size, data }
    if (val && typeof val === "object") {
      const obj = val as Record<string, unknown>;
      if (typeof obj.name === "string") {
        return getFileLabel(
          obj.name,
          typeof obj.type === "string" ? obj.type : "",
        );
      }
      if (typeof obj.type === "string") {
        const mimeParts = obj.type.split("/");
        if (mimeParts.length > 1) {
          return `${mimeParts[1].toUpperCase()} file`;
        }
        return `${obj.type} file`;
      }
      return "File";
    }

    // Handle string values (workspace URIs, data URIs, or file paths)
    if (typeof val !== "string") {
      return "File";
    }

    const wsURI = parseWorkspaceURI(val);
    if (wsURI) {
      if (wsURI.mimeType) {
        const parts = wsURI.mimeType.split("/");
        if (parts.length > 1) {
          return `${parts[1].toUpperCase()} file`;
        }
        return "File";
      }
      return "Uploaded file";
    }

    if (val.startsWith("data:")) {
      const matches = val.match(/^data:([^;]+);/);
      if (matches?.[1]) {
        const mimeParts = matches[1].split("/");
        if (mimeParts.length > 1) {
          return `${mimeParts[1].toUpperCase()} file`;
        }
        return `${matches[1]} file`;
      }
    } else {
      const pathParts = val.split(".");
      if (pathParts.length > 1) {
        const ext = pathParts.pop();
        if (ext) return `${ext.toUpperCase()} file`;
      }
    }
    return "File";
  }

  function processFileBase64(file: File) {
    setIsUploading(true);
    setUploadError(null);

    const reader = new FileReader();
    reader.onload = (e) => {
      const base64String = e.target?.result as string;
      setFileInfo({
        name: file.name,
        size: file.size,
        content_type: file.type || "application/octet-stream",
      });
      onChange(base64String);
      setIsUploading(false);
    };
    reader.onerror = () => {
      setUploadError("Failed to read file");
      setIsUploading(false);
    };
    reader.readAsDataURL(file);
  }

  async function uploadFile(file: File) {
    if (mode === "base64") {
      processFileBase64(file);
      return;
    }

    if (!onUploadFile) {
      setUploadError("Upload handler not provided");
      return;
    }

    setIsUploading(true);
    setUploadError(null);

    const oldURI = value;

    try {
      const result = await onUploadFile(file);

      setFileInfo({
        name: result.file_name,
        size: result.size,
        content_type: result.content_type,
      });

      onChange(result.file_uri);

      // Delete the old file only after the new upload succeeds
      if (oldURI && onDeleteFile) {
        onDeleteFile(oldURI);
      }
    } catch (error) {
      console.error("Upload failed:", error);
      setUploadError(error instanceof Error ? error.message : "Upload failed");
    } finally {
      setIsUploading(false);
    }
  }

  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    // Validate max size
    if (typeof maxFileSize === "number" && file.size > maxFileSize) {
      setUploadError(
        `File exceeds maximum size of ${formatFileSize(maxFileSize)} (selected ${formatFileSize(file.size)})`,
      );
      return;
    }
    // Validate accept types
    if (!isAcceptedType(file, accept)) {
      setUploadError("Selected file type is not allowed");
      return;
    }
    uploadFile(file);
  }

  function handleFileDrop(event: React.DragEvent<HTMLDivElement>) {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) uploadFile(file);
  }

  function handleClear() {
    if (value && onDeleteFile) {
      onDeleteFile(value);
    }
    if (inputRef.current) {
      inputRef.current.value = "";
    }
    onChange("");
    setFileInfo(null);
  }

  const displayName = placeholder || "File";

  if (variant === "compact") {
    return (
      <div
        data-slot="file-input"
        className={cn("flex flex-col gap-1.5", className)}
      >
        <div
          data-slot="file-input-body"
          className="nodrag flex flex-col gap-1.5"
        >
          {isUploading ? (
            <div
              data-slot="file-input-progress"
              className="flex items-center gap-2 rounded-md border border-blue-200 bg-info-surface p-2 dark:border-blue-800 dark:bg-blue-950"
            >
              <div
                data-slot="file-input-spinner"
                className="h-4 w-4 animate-spin rounded-full border-2 border-blue-300 border-t-info-foreground dark:border-blue-700 dark:border-t-blue-400"
              />
              <Text
                data-slot="file-input-progress-text"
                variant="small"
                className="text-blue-700 dark:text-blue-300"
              >
                {mode === "base64" ? "Processing..." : "Uploading..."}
              </Text>
            </div>
          ) : value ? (
            <div
              data-slot="file-input-file-row"
              className="flex items-center gap-2"
            >
              <div
                data-slot="file-input-file"
                className="flex flex-1 items-center gap-2 rounded-xlarge border border-gray-300 bg-gray-50 p-2 dark:border-gray-600 dark:bg-gray-800"
              >
                <Icon
                  data-slot="file-input-file-icon"
                  icon={File02Icon}
                  className="h-4 w-4 flex-shrink-0 text-gray-600 dark:text-gray-400"
                />

                <Text
                  data-slot="file-input-file-name"
                  variant="small-medium"
                  className="truncate text-gray-900 dark:text-gray-100"
                >
                  {fileInfo
                    ? getFileLabel(fileInfo.name, fileInfo.content_type)
                    : getFileLabelFromValue(value)}
                </Text>
                {fileInfo && (
                  <Text
                    data-slot="file-input-file-size"
                    variant="small"
                    className="text-gray-500 dark:text-gray-400"
                  >
                    {formatFileSize(fileInfo.size)}
                  </Text>
                )}
              </div>
              {isPreviewableFile(value, fileInfo?.content_type) && (
                <PreviewButton
                  value={value}
                  title={
                    fileInfo
                      ? getFileLabel(fileInfo.name, fileInfo.content_type)
                      : "Preview"
                  }
                  contentType={fileInfo?.content_type}
                />
              )}
              <Button
                data-slot="file-input-clear"
                variant="outline"
                size="small"
                className="h-7 w-7 min-w-0 flex-shrink-0 border-border-strong p-0 text-gray-500 hover:text-error-hover dark:text-gray-400 dark:hover:text-red-500"
                onClick={handleClear}
                type="button"
                aria-label="Clear file"
              >
                <Icon
                  data-slot="file-input-clear-icon"
                  icon={Cancel01Icon}
                  size={14}
                />
              </Button>
            </div>
          ) : (
            <div
              data-slot="file-input-upload-row"
              className="flex items-center gap-2"
            >
              <Button
                data-slot="file-input-upload"
                variant="outline"
                size="small"
                onClick={() => inputRef.current?.click()}
                className="flex-1 border-border-strong text-caption leading-4"
                disabled={isUploading}
                type="button"
              >
                <Icon
                  data-slot="file-input-upload-icon"
                  icon={Upload01Icon}
                  className="mr-1.5 h-3.5 w-3.5"
                />
                {`Upload ${displayName}`}
              </Button>
            </div>
          )}
          <input
            ref={inputRef}
            data-slot="file-input-control"
            type="file"
            accept={acceptToString(accept)}
            onChange={handleFileChange}
            className="hidden"
            disabled={isUploading}
          />
        </div>
        {uploadError && (
          <Text
            data-slot="file-input-error"
            variant="small"
            tone="danger"
            className="dark:text-red-400"
          >
            {uploadError}
          </Text>
        )}
      </div>
    );
  }

  return (
    <div data-slot="file-input" className={cn("w-full", className)}>
      {isUploading ? (
        <div data-slot="file-input-section" className="space-y-2">
          <div
            data-slot="file-input-row"
            className="flex min-h-14 items-center gap-4"
          >
            <div
              data-slot="file-input-progress"
              className="agpt-border-input flex min-h-14 w-full items-center gap-3 rounded-xl bg-surface-muted p-4 text-body leading-5"
            >
              <div
                data-slot="file-input-spinner"
                className="h-5 w-5 animate-spin rounded-full border-2 border-blue-300 border-t-info-foreground"
              />
              <span
                data-slot="file-input-progress-text"
                className="text-gray-700"
              >
                {mode === "base64" ? "Processing..." : "Uploading..."}
              </span>
            </div>
          </div>
        </div>
      ) : value ? (
        <div data-slot="file-input-section" className="space-y-2">
          <div
            data-slot="file-input-row"
            className="flex min-h-14 items-center gap-4"
          >
            <div
              data-slot="file-input-file"
              className="agpt-border-input flex min-h-14 w-full items-center justify-between rounded-xl bg-surface-muted p-4 text-body leading-5 text-gray-500"
            >
              <div
                data-slot="file-input-file-info"
                className="flex items-center gap-2"
              >
                <Icon
                  data-slot="file-input-file-icon"
                  icon={File02Icon}
                  className="h-7 w-7 text-foreground-base"
                />
                <div
                  data-slot="file-input-file-details"
                  className="flex flex-col gap-0.5"
                >
                  <span
                    data-slot="file-input-file-name"
                    className="font-normal text-foreground-base"
                  >
                    {fileInfo
                      ? getFileLabel(fileInfo.name, fileInfo.content_type)
                      : getFileLabelFromValue(value)}
                  </span>
                  <span data-slot="file-input-file-size">
                    {fileInfo ? formatFileSize(fileInfo.size) : ""}
                  </span>
                </div>
              </div>
              <div
                data-slot="file-input-actions"
                className="flex items-center gap-2"
              >
                {isPreviewableFile(value, fileInfo?.content_type) && (
                  <PreviewButton
                    value={value}
                    title={
                      fileInfo
                        ? getFileLabel(fileInfo.name, fileInfo.content_type)
                        : "Preview"
                    }
                    contentType={fileInfo?.content_type}
                  />
                )}
                <Button
                  data-slot="file-input-clear"
                  variant="outline"
                  size="small"
                  type="button"
                  onClick={handleClear}
                  aria-label="Clear file"
                  className="h-7 w-7 min-w-0 flex-shrink-0 border-border-strong p-0 text-gray-500 hover:text-error-hover dark:text-gray-400 dark:hover:text-red-500"
                >
                  <Icon
                    data-slot="file-input-clear-icon"
                    icon={Delete02Icon}
                    className="h-5 w-5"
                  />
                </Button>
              </div>
            </div>
          </div>
          {showStorageNote && mode === "upload" && (
            <p
              data-slot="file-input-note"
              className="text-caption leading-4 text-gray-500"
            >
              {storageNote}
            </p>
          )}
        </div>
      ) : (
        <div data-slot="file-input-section" className="space-y-2">
          <div
            data-slot="file-input-row"
            className="flex min-h-14 items-center gap-4"
          >
            <div
              data-slot="file-input-dropzone"
              onDrop={handleFileDrop}
              onDragOver={(e) => e.preventDefault()}
              className="agpt-border-input flex min-h-14 w-full items-center justify-center rounded-xl border-dashed bg-surface-muted text-body leading-5 text-gray-500"
            >
              Choose a file or drag and drop it here
            </div>

            <Button
              data-slot="file-input-browse"
              onClick={() => inputRef.current?.click()}
              className="min-w-40"
              type="button"
            >
              Browse File
            </Button>
          </div>

          {uploadError && (
            <div
              data-slot="file-input-error"
              className="text-body leading-5 text-error-hover"
            >
              Error: {uploadError}
            </div>
          )}

          {showStorageNote && mode === "upload" && (
            <p
              data-slot="file-input-note"
              className="text-caption leading-4 text-gray-500"
            >
              {storageNote}
            </p>
          )}
        </div>
      )}

      <input
        ref={inputRef}
        data-slot="file-input-control"
        type="file"
        accept={acceptToString(accept)}
        className="hidden"
        onChange={handleFileChange}
        disabled={isUploading}
      />
    </div>
  );
}
