"use client";

import { extendedButtonVariants } from "@/components/atoms/Button/helpers";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { cn } from "@/lib/utils";
import {
  ArrowDown01Icon,
  FolderAddIcon,
  Loading03Icon,
  Upload03Icon,
} from "@hugeicons/core-free-icons";
import { FolderFormDialog } from "../WorkspaceFolders/FolderFormDialog";
import { useNewMenu } from "./useNewMenu";

interface Props {
  selectedFolderId: string | null;
}

export function NewMenu({ selectedFolderId }: Props) {
  const {
    fileInputRef,
    isUploading,
    openFilePicker,
    handleFilesSelected,
    isCreateOpen,
    setIsCreateOpen,
    isCreating,
    handleCreateFolder,
  } = useNewMenu(selectedFolderId);

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button
            type="button"
            disabled={isUploading}
            className={cn(
              extendedButtonVariants({ variant: "primary", size: "small" }),
              "min-w-0 gap-1.5 pl-4 pr-3",
            )}
            data-testid="artifacts-new-menu"
          >
            {isUploading ? (
              <Icon icon={Loading03Icon} size={16} className="animate-spin" />
            ) : null}
            {isUploading ? "Uploading…" : "New"}
            <Icon icon={ArrowDown01Icon} size={16} />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-44">
          <DropdownMenuItem
            onSelect={openFilePicker}
            data-testid="artifacts-upload-file"
          >
            <Icon icon={Upload03Icon} size={16} className="mr-2" />
            Upload file
          </DropdownMenuItem>
          <DropdownMenuItem
            onSelect={() => setIsCreateOpen(true)}
            data-testid="create-folder-button"
          >
            <Icon icon={FolderAddIcon} size={16} className="mr-2" />
            New folder
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <input
        ref={fileInputRef}
        type="file"
        multiple
        className="hidden"
        tabIndex={-1}
        aria-hidden
        onChange={handleFilesSelected}
        data-testid="artifacts-upload-input"
      />
      <FolderFormDialog
        isOpen={isCreateOpen}
        setIsOpen={setIsCreateOpen}
        mode="create"
        isSubmitting={isCreating}
        onSubmit={handleCreateFolder}
      />
    </>
  );
}
