"use client";

import { Button } from "@/components/atoms/Button/Button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { cn } from "@/lib/utils";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import {
  BookOpenIcon,
  CalendarDotsIcon,
  FolderOpenIcon,
  PaperclipIcon,
  PlugsConnectedIcon,
  PlusIcon,
} from "@phosphor-icons/react";
import { useRef } from "react";
import { useCopilotModal } from "../../../useCopilotModal";

interface Props {
  onFilesSelected: (files: File[]) => void;
  onUseWorkspaceFile?: () => void;
  disabled?: boolean;
}

export function ComposerPlusMenu({
  onFilesSelected,
  onUseWorkspaceFile,
  disabled,
}: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { openModal } = useCopilotModal();
  const showWorkspaceOption = useGetFlag(Flag.CHAT_WORKSPACE_FILES);

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
            aria-label="Open composer menu"
            data-testid="composer-plus-button"
            disabled={disabled}
            className={cn(
              "border-zinc-300 bg-white text-zinc-500 hover:border-zinc-400 hover:bg-zinc-50 hover:text-zinc-700",
              disabled && "opacity-40",
            )}
          >
            <PlusIcon className="h-4 w-4" weight="bold" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="min-w-[14rem]">
          <DropdownMenuItem onSelect={openFilePicker}>
            <PaperclipIcon className="mr-2 h-4 w-4" />
            Attach file
          </DropdownMenuItem>
          {showWorkspaceOption && (
            <DropdownMenuItem onSelect={() => onUseWorkspaceFile?.()}>
              <FolderOpenIcon className="mr-2 h-4 w-4" />
              Use File from Workspace
            </DropdownMenuItem>
          )}
          <DropdownMenuItem onSelect={() => openModal("integrations")}>
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
