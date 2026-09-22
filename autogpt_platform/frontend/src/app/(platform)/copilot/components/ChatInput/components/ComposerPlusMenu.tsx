"use client";

import { Button } from "@/components/atoms/Button/Button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { cn } from "@/lib/utils";
import { useComposerPlusMenu } from "./useComposerPlusMenu";
import {
  Attachment01Icon,
  BookOpen01Icon,
  Calendar03Icon,
  FolderOpenIcon,
  PlugSocketIcon,
  PlusSignIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  onFilesSelected: (files: File[]) => void;
  onUseWorkspaceFile?: () => void;
  onClearGuidedPrompt?: () => void;
  disabled?: boolean;
  className?: string;
}

export function ComposerPlusMenu({
  onFilesSelected,
  onUseWorkspaceFile,
  onClearGuidedPrompt,
  disabled,
  className,
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
              "border-transparent bg-transparent text-black shadow-none hover:border-transparent hover:bg-zinc-100 hover:text-black",
              className,
              disabled && "opacity-40",
            )}
          >
            <Icon icon={PlusSignIcon} className="h-4 w-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="min-w-[14rem]">
          <DropdownMenuItem
            onSelect={() => {
              onClearGuidedPrompt?.();
              openFilePicker();
            }}
          >
            <Icon icon={Attachment01Icon} className="mr-2 h-4 w-4" />
            Attach file
          </DropdownMenuItem>
          {showWorkspaceOption && (
            <DropdownMenuItem
              onSelect={() => {
                onClearGuidedPrompt?.();
                onUseWorkspaceFile?.();
              }}
            >
              <Icon icon={FolderOpenIcon} className="mr-2 h-4 w-4" />
              Use File from Workspace
            </DropdownMenuItem>
          )}
          <DropdownMenuItem
            onSelect={() => {
              onClearGuidedPrompt?.();
              openModal("connect");
            }}
          >
            <Icon icon={PlugSocketIcon} className="mr-2 h-4 w-4" />
            Connect service
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={() => openModal("skills")}>
            <Icon icon={BookOpen01Icon} className="mr-2 h-4 w-4" />
            Skills
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={() => openModal("scheduled")}>
            <Icon icon={Calendar03Icon} className="mr-2 h-4 w-4" />
            Scheduled
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </>
  );
}
