import { FileTextIcon, TrashIcon, UploadIcon } from "@phosphor-icons/react";
import { Cross2Icon } from "@radix-ui/react-icons";
import { useRef, useState } from "react";
import { Button } from "../Button/Button";
import { formatFileSize, getFileLabel } from "./helpers";
import { cn } from "@/lib/utils";
import { Progress } from "../Progress/Progress";
import { Text } from "../Text/Text";

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
  className?: string;
  maxFileSize?: number;
  accept?: string | string[];
  variant?: FileInputVariant;
  showStorageNote?: boolean;
}

interface UploadModeProps extends BaseProps {
  mode?: "upload";
  onUploadFile: (file: File) => Promise<UploadFileResult>;
  uploadProgress: number;
}

interface Base64ModeProps extends BaseProps {
  mode: "base64";
  onUploadFile?: never;
  uploadProgress?: never;
}

type Props = UploadModeProps | Base64ModeProps;

export function FileInput(props: Props) {
  const {
    value,
    onChange,
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
  const uploadProgress =
    mode === "upload" ? (props as UploadModeProps).uploadProgress : 0;

  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [fileInfo, setFileInfo] = useState<{
    name: string;
    size: number;
    content_type: string;
  } | null>(null);

  const inputRef = useRef<HTMLInputElement>(null);

  const storageNote =
    "Files are stored securely and will be automatically deleted at most 24 hours after upload.";

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

  const getFileLabelFromValue = (val: unknown): string => {
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

    // Handle string values (data URIs or file paths)
    if (typeof val !== "string") {
      return "File";
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
  };

  const processFileBase64 = (file: File) => {
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
  };

  const uploadFile = async (file: File) => {
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

    try {
      const result = await onUploadFile(file);

      setFileInfo({
        name: result.file_name,
        size: result.size,
        content_type: result.content_type,
      });

      onChange(result.file_uri);
    } catch (error) {
      console.error("Upload failed:", error);
      setUploadError(error instanceof Error ? error.message : "Upload failed");
    } finally {
      setIsUploading(false);
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
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
  };

  const handleFileDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) uploadFile(file);
  };

  const handleClear = () => {
    if (inputRef.current) {
      inputRef.current.value = "";
    }
    onChange("");
    setFileInfo(null);
  };

  const displayName = placeholder || "File";

  if (variant === "compact") {
    return (
      <div className={cn("flex flex-col gap-1.5", className)}>
        <div className="nodrag flex flex-col gap-1.5">
          {isUploading ? (
            <div className="flex flex-col gap-1.5 rounded-md border border-blue-200 bg-blue-50 p-2 dark:border-blue-800 dark:bg-blue-950">
              <div className="flex items-center gap-2">
                <UploadIcon className="h-4 w-4 animate-pulse text-blue-600 dark:text-blue-400" />
                <Text
                  variant="small"
                  className="text-blue-700 dark:text-blue-300"
                >
                  {mode === "base64" ? "Processing..." : "Uploading..."}
                </Text>
                {mode === "upload" && (
                  <Text
                    variant="small-medium"
                    className="ml-auto text-blue-600 dark:text-blue-400"
                  >
                    {Math.round(uploadProgress)}%
                  </Text>
                )}
              </div>
              {mode === "upload" && (
                <Progress value={uploadProgress} className="h-1 w-full" />
              )}
            </div>
          ) : value ? (
            <div className="flex items-center gap-2">
              <div className="flex flex-1 items-center gap-2 rounded-xlarge border border-gray-300 bg-gray-50 p-2 dark:border-gray-600 dark:bg-gray-800">
                <FileTextIcon className="h-4 w-4 flex-shrink-0 text-gray-600 dark:text-gray-400" />

                <Text
                  variant="small-medium"
                  className="truncate text-gray-900 dark:text-gray-100"
                >
                  {fileInfo
                    ? getFileLabel(fileInfo.name, fileInfo.content_type)
                    : getFileLabelFromValue(value)}
                </Text>
                {fileInfo && (
                  <Text
                    variant="small"
                    className="text-gray-500 dark:text-gray-400"
                  >
                    {formatFileSize(fileInfo.size)}
                  </Text>
                )}
              </div>
              <Button
                variant="outline"
                size="small"
                className="h-7 w-7 min-w-0 flex-shrink-0 border-zinc-300 p-0 text-gray-500 hover:text-red-600 dark:text-gray-400 dark:hover:text-red-500"
                onClick={handleClear}
                type="button"
              >
                <Cross2Icon className="h-3.5 w-3.5" />
              </Button>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="small"
                onClick={() => inputRef.current?.click()}
                className="flex-1 border-zinc-300 text-xs"
                disabled={isUploading}
                type="button"
              >
                <UploadIcon className="mr-1.5 h-3.5 w-3.5" />
                {`Upload ${displayName}`}
              </Button>
            </div>
          )}
          <input
            ref={inputRef}
            type="file"
            accept={acceptToString(accept)}
            onChange={handleFileChange}
            className="hidden"
            disabled={isUploading}
          />
        </div>
        {uploadError && (
          <Text variant="small" className="text-red-600 dark:text-red-400">
            {uploadError}
          </Text>
        )}
      </div>
    );
  }

  return (
    <div className={cn("w-full", className)}>
      {isUploading ? (
        <div className="space-y-2">
          <div className="flex min-h-14 items-center gap-4">
            <div className="agpt-border-input flex min-h-14 w-full flex-col justify-center rounded-xl bg-zinc-50 p-4 text-sm">
              <div className="mb-2 flex items-center gap-2">
                <UploadIcon className="h-5 w-5 text-blue-600" />
                <span className="text-gray-700">
                  {mode === "base64" ? "Processing..." : "Uploading..."}
                </span>
                {mode === "upload" && (
                  <span className="text-gray-500">
                    {Math.round(uploadProgress)}%
                  </span>
                )}
              </div>
              {mode === "upload" && (
                <Progress value={uploadProgress} className="w-full" />
              )}
            </div>
          </div>
          {showStorageNote && mode === "upload" && (
            <p className="text-xs text-gray-500">{storageNote}</p>
          )}
        </div>
      ) : value ? (
        <div className="space-y-2">
          <div className="flex min-h-14 items-center gap-4">
            <div className="agpt-border-input flex min-h-14 w-full items-center justify-between rounded-xl bg-zinc-50 p-4 text-sm text-gray-500">
              <div className="flex items-center gap-2">
                <FileTextIcon className="h-7 w-7 text-black" />
                <div className="flex flex-col gap-0.5">
                  <span className="font-normal text-black">
                    {fileInfo
                      ? getFileLabel(fileInfo.name, fileInfo.content_type)
                      : getFileLabelFromValue(value)}
                  </span>
                  <span>{fileInfo ? formatFileSize(fileInfo.size) : ""}</span>
                </div>
              </div>
              <TrashIcon
                className="h-5 w-5 cursor-pointer text-black"
                onClick={handleClear}
              />
            </div>
          </div>
          {showStorageNote && mode === "upload" && (
            <p className="text-xs text-gray-500">{storageNote}</p>
          )}
        </div>
      ) : (
        <div className="space-y-2">
          <div className="flex min-h-14 items-center gap-4">
            <div
              onDrop={handleFileDrop}
              onDragOver={(e) => e.preventDefault()}
              className="agpt-border-input flex min-h-14 w-full items-center justify-center rounded-xl border-dashed bg-zinc-50 text-sm text-gray-500"
            >
              Choose a file or drag and drop it here
            </div>

            <Button
              onClick={() => inputRef.current?.click()}
              className="min-w-40"
              type="button"
            >
              Browse File
            </Button>
          </div>

          {uploadError && (
            <div className="text-sm text-red-600">Error: {uploadError}</div>
          )}

          {showStorageNote && mode === "upload" && (
            <p className="text-xs text-gray-500">{storageNote}</p>
          )}
        </div>
      )}

      <input
        ref={inputRef}
        type="file"
        accept={acceptToString(accept)}
        className="hidden"
        onChange={handleFileChange}
        disabled={isUploading}
      />
    </div>
  );
}
