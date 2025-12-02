import { Button } from "@/components/atoms/Button/Button";
import type { GoogleDrivePickerConfig } from "@/lib/autogpt-server-api/types";
import { cn } from "@/lib/utils";
import { Cross2Icon } from "@radix-ui/react-icons";
import React, { useCallback } from "react";
import { GoogleDrivePicker } from "./GoogleDrivePicker";

export interface GoogleDrivePickerInputProps {
  config: GoogleDrivePickerConfig;
  value: any;
  onChange: (value: any) => void;
  error?: string;
  className?: string;
  showRemoveButton?: boolean;
}

export function GoogleDrivePickerInput({
  config,
  value,
  onChange,
  error,
  className,
  showRemoveButton = true,
}: GoogleDrivePickerInputProps) {
  const [pickerError, setPickerError] = React.useState<string | null>(null);
  const isMultiSelect = config.multiselect || false;
  const currentFiles = isMultiSelect
    ? Array.isArray(value)
      ? value
      : []
    : value
      ? [value]
      : [];

  const handlePicked = useCallback(
    (files: any[]) => {
      // Clear any previous picker errors
      setPickerError(null);

      // Convert to GoogleDriveFile format
      const convertedFiles = files.map((f) => ({
        id: f.id,
        name: f.name,
        mimeType: f.mimeType,
        url: f.url,
        iconUrl: f.iconUrl,
        isFolder: f.mimeType === "application/vnd.google-apps.folder",
      }));

      // Store based on multiselect mode
      const newValue = isMultiSelect ? convertedFiles : convertedFiles[0];
      onChange(newValue);
    },
    [isMultiSelect, onChange],
  );

  const handleRemoveFile = useCallback(
    (idx: number) => {
      if (isMultiSelect) {
        const newFiles = currentFiles.filter((_: any, i: number) => i !== idx);
        onChange(newFiles);
      } else {
        onChange(null);
      }
    },
    [isMultiSelect, currentFiles, onChange],
  );

  const handleError = useCallback((error: any) => {
    console.error("Google Drive Picker error:", error);
    setPickerError(error instanceof Error ? error.message : String(error));
  }, []);

  return (
    <div className={cn("flex flex-col gap-2", className)}>
      {/* Picker Button */}
      <GoogleDrivePicker
        multiselect={config.multiselect || false}
        views={config.allowed_views || ["DOCS"]}
        scopes={config.scopes || ["https://www.googleapis.com/auth/drive.file"]}
        disabled={false}
        onPicked={handlePicked}
        onCanceled={() => {
          // User canceled - no action needed
        }}
        onError={handleError}
      />

      {/* Display Selected Files */}
      {currentFiles.length > 0 && (
        <div className="space-y-1">
          {currentFiles.map((file: any, idx: number) => (
            <div
              key={file.id || idx}
              className={cn(
                "flex items-center gap-2",
                showRemoveButton
                  ? "justify-between rounded-md border border-gray-300 bg-gray-50 px-3 py-2 text-sm dark:border-gray-600 dark:bg-gray-800"
                  : "text-sm text-gray-600 dark:text-gray-400",
              )}
            >
              <div className="flex items-center gap-2 overflow-hidden">
                {file.iconUrl && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={file.iconUrl}
                    alt=""
                    className="h-4 w-4 flex-shrink-0"
                  />
                )}
                <span className="truncate" title={file.name}>
                  {file.name || file.id}
                </span>
              </div>

              {showRemoveButton && (
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-6 w-6 flex-shrink-0"
                  onClick={() => handleRemoveFile(idx)}
                >
                  <Cross2Icon className="h-3 w-3" />
                </Button>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Error Messages */}
      {error && <span className="text-sm text-red-500">{error}</span>}
      {pickerError && (
        <span className="text-sm text-red-500">{pickerError}</span>
      )}
    </div>
  );
}
