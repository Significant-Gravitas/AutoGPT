"use client";

import { Button } from "@/components/atoms/Button/Button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { cn } from "@/lib/utils";
import {
  FolderOpen as FolderOpenIcon,
  Plus as PlusIcon,
  UploadSimple as UploadSimpleIcon,
} from "@phosphor-icons/react";
import { useRef } from "react";

interface Props {
  onFilesSelected: (files: File[]) => void;
  onUseWorkspaceFile?: () => void;
  showWorkspaceOption?: boolean;
  disabled?: boolean;
}

export function AttachmentMenu({
  onFilesSelected,
  onUseWorkspaceFile,
  showWorkspaceOption = false,
  disabled,
}: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  function openFilePicker() {
    fileInputRef.current?.click();
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    if (files.length > 0) {
      onFilesSelected(files);
    }
    // Reset so the same file can be re-selected
    e.target.value = "";
  }

  const buttonClassName = cn(
    "border-zinc-300 bg-white text-zinc-500 hover:border-zinc-400 hover:bg-zinc-50 hover:text-zinc-700",
    disabled && "opacity-40",
  );

  return (
    <>
      <input
        ref={fileInputRef}
        type="file"
        multiple
        className="hidden"
        onChange={handleFileChange}
        tabIndex={-1}
      />
      {showWorkspaceOption ? (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              type="button"
              variant="icon"
              size="icon"
              aria-label="Attach file"
              disabled={disabled}
              className={buttonClassName}
            >
              <PlusIcon className="h-4 w-4" weight="bold" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="min-w-[14rem]">
            <DropdownMenuItem onSelect={openFilePicker}>
              <UploadSimpleIcon className="mr-2 h-4 w-4" />
              Upload from Computer
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={() => onUseWorkspaceFile?.()}>
              <FolderOpenIcon className="mr-2 h-4 w-4" />
              Use File from Workspace
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      ) : (
        <Button
          type="button"
          variant="icon"
          size="icon"
          aria-label="Attach file"
          disabled={disabled}
          onClick={openFilePicker}
          className={buttonClassName}
        >
          <PlusIcon className="h-4 w-4" weight="bold" />
        </Button>
      )}
    </>
  );
}
