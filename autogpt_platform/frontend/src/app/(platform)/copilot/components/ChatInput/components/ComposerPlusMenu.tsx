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
  BookOpenIcon,
  CalendarDotsIcon,
  FolderOpenIcon,
  PaperclipIcon,
  PlugsConnectedIcon,
  PlusIcon,
} from "@phosphor-icons/react";
import { useComposerPlusMenu } from "./useComposerPlusMenu";

interface Props {
  onFilesSelected: (files: File[]) => void;
  onUseWorkspaceFile?: () => void;
  onClearGuidedPrompt?: () => void;
  disabled?: boolean;
}

export function ComposerPlusMenu({
  onFilesSelected,
  onUseWorkspaceFile,
  onClearGuidedPrompt,
  disabled,
}: Props) {
  const {
    fileInputRef,
    openModal,
    showWorkspaceOption,
    openFilePicker,
    handleFileChange,
  } = useComposerPlusMenu({ onFilesSelected });

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
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            type="button"
            variant="icon"
            size="icon"
            aria-label="Add files and more"
            data-testid="composer-plus-button"
            disabled={disabled}
            className={cn(
              "border-neutral-200 bg-white text-zinc-500 shadow-sm hover:border-neutral-200 hover:bg-neutral-50 hover:text-zinc-700",
              disabled && "opacity-40",
            )}
          >
            <PlusIcon className="h-4 w-4" weight="bold" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="min-w-[14rem]">
          <DropdownMenuItem
            onSelect={() => {
              onClearGuidedPrompt?.();
              openFilePicker();
            }}
          >
            <PaperclipIcon className="mr-2 h-4 w-4" />
            Attach file
          </DropdownMenuItem>
          {showWorkspaceOption && (
            <DropdownMenuItem
              onSelect={() => {
                onClearGuidedPrompt?.();
                onUseWorkspaceFile?.();
              }}
            >
              <FolderOpenIcon className="mr-2 h-4 w-4" />
              Use File from Workspace
            </DropdownMenuItem>
          )}
          <DropdownMenuItem
            onSelect={() => {
              onClearGuidedPrompt?.();
              openModal("integrations");
            }}
          >
            <PlugsConnectedIcon className="mr-2 h-4 w-4" />
            Integrations
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={() => openModal("skills")}>
            <BookOpenIcon className="mr-2 h-4 w-4" />
            Skills
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={() => openModal("scheduled")}>
            <CalendarDotsIcon className="mr-2 h-4 w-4" />
            Scheduled
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </>
  );
}
