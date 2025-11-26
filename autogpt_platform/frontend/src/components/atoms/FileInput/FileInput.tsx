import { FileTextIcon, TrashIcon, UploadIcon } from "@phosphor-icons/react";
import { useRef, useState } from "react";
import { Button } from "../Button/Button";
import { formatFileSize, getFileLabel } from "./helpers";
import { cn } from "@/lib/utils";
import { Progress } from "../Progress/Progress";

type UploadFileResult = {
  file_name: string;
  size: number;
  content_type: string;
  file_uri: string;
};

interface Props {
  onUploadFile: (file: File) => Promise<UploadFileResult>;
  uploadProgress: number;
  value?: string; // file URI or empty
  placeholder?: string; // e.g. "Resume", "Document", etc.
  onChange: (value: string) => void;
  className?: string;
  maxFileSize?: number; // bytes (optional)
  accept?: string | string[]; // input accept filter (optional)
}

export function FileInput({
  onUploadFile,
  uploadProgress,
  value,
  onChange,
  className,
  maxFileSize,
  accept,
}: Props) {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [fileInfo, setFileInfo] = useState<{
    name: string;
    size: number;
    content_type: string;
  } | null>(null);

  const uploadFile = async (file: File) => {
    setIsUploading(true);
    setUploadError(null);

    try {
      const result = await onUploadFile(file);

      setFileInfo({
        name: result.file_name,
        size: result.size,
        content_type: result.content_type,
      });

      // Set the file URI as the value
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
    const fileType = file.type; // e.g. image/png
    const fileExt = file.name.includes(".")
      ? `.${file.name.split(".").pop()}`.toLowerCase()
      : "";

    for (const entry of list) {
      if (!entry) continue;
      const e = entry.toLowerCase();
      if (e.includes("/")) {
        // MIME type, support wildcards like image/*
        const [main, sub] = e.split("/");
        const [fMain, fSub] = fileType.toLowerCase().split("/");
        if (!fMain || !fSub) continue;
        if (sub === "*") {
          if (main === fMain) return true;
        } else {
          if (e === fileType.toLowerCase()) return true;
        }
      } else if (e.startsWith(".")) {
        // Extension match
        if (fileExt === e) return true;
      }
    }
    return false;
  }

  return (
    <div className={cn("w-full", className)}>
      {isUploading ? (
        <div className="space-y-2">
          <div className="flex min-h-14 items-center gap-4">
            <div className="agpt-border-input flex min-h-14 w-full flex-col justify-center rounded-xl bg-zinc-50 p-4 text-sm">
              <div className="mb-2 flex items-center gap-2">
                <UploadIcon className="h-5 w-5 text-blue-600" />
                <span className="text-gray-700">Uploading...</span>
                <span className="text-gray-500">
                  {Math.round(uploadProgress)}%
                </span>
              </div>
              <Progress value={uploadProgress} className="w-full" />
            </div>
          </div>
          <p className="text-xs text-gray-500">{storageNote}</p>
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
                      : "File"}
                  </span>
                  <span>{fileInfo ? formatFileSize(fileInfo.size) : ""}</span>
                </div>
              </div>
              <TrashIcon
                className="h-5 w-5 cursor-pointer text-black"
                onClick={() => {
                  if (inputRef.current) {
                    inputRef.current.value = "";
                  }
                  onChange("");
                  setFileInfo(null);
                }}
              />
            </div>
          </div>
          <p className="text-xs text-gray-500">{storageNote}</p>
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
            >
              Browse File
            </Button>
          </div>

          {uploadError && (
            <div className="text-sm text-red-600">Error: {uploadError}</div>
          )}

          <p className="text-xs text-gray-500">{storageNote}</p>
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
