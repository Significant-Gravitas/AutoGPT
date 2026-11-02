"use client";

import { Button } from "@/components/atoms/Button/Button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import type { IconSvgElement } from "@hugeicons/react";
import { cn } from "@/lib/utils";
import { MAX_ATTACHMENTS } from "../../../helpers/workspaceAttachments";
import { useComposerPlusMenu } from "./useComposerPlusMenu";
import {
  Attachment01Icon,
  BookOpen01Icon,
  Calendar03Icon,
  Camera01Icon,
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
  /** The message already holds `MAX_ATTACHMENTS` attachments: both file
   *  entries read disabled rather than letting the composer take another. */
  isAtCap?: boolean;
  className?: string;
  /** Composer settings that live behind the + rather than as icons in the
   *  tray (voice mode, model tier, the dev token readout). */
  extraItems?: ComposerMenuItem[];
}

export interface ComposerMenuItem {
  key: string;
  label: string;
  icon: IconSvgElement;
  onSelect: () => void;
  disabled?: boolean;
}

export function ComposerPlusMenu({
  onFilesSelected,
  onUseWorkspaceFile,
  onClearGuidedPrompt,
  disabled,
  isAtCap = false,
  className,
  extraItems = [],
}: Props) {
  const {
    fileInputRef,
    openModal,
    cameraInputRef,
    showWorkspaceOption,
    showCameraOption,
    openFilePicker,
    openCamera,
    handleFileChange,
    handlePhotoChange,
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
      {/* Opens the rear camera on phones and tablets; the captured photo
          goes through the same path as any attached file. */}
      <input
        ref={cameraInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        className="hidden"
        onChange={handlePhotoChange}
        tabIndex={-1}
        data-testid="composer-camera-input"
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
              "border-transparent bg-transparent text-foreground-base shadow-none hover:border-transparent hover:bg-surface-hover hover:text-foreground-base",
              className,
              disabled && "opacity-40",
            )}
          >
            <Icon icon={PlusSignIcon} className="h-4 w-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="min-w-[14rem]">
          <DropdownMenuItem
            disabled={isAtCap}
            onSelect={() => {
              onClearGuidedPrompt?.();
              openFilePicker();
            }}
          >
            <Icon icon={Attachment01Icon} className="mr-2 h-4 w-4" />
            Attach file
            {isAtCap && <CapLabel />}
          </DropdownMenuItem>
          {showCameraOption && (
            <DropdownMenuItem
              disabled={isAtCap}
              onSelect={() => {
                onClearGuidedPrompt?.();
                openCamera();
              }}
            >
              <Icon icon={Camera01Icon} className="mr-2 h-4 w-4" />
              Take photo
              {isAtCap && <CapLabel />}
            </DropdownMenuItem>
          )}
          {showWorkspaceOption && (
            <DropdownMenuItem
              disabled={isAtCap}
              onSelect={() => {
                onClearGuidedPrompt?.();
                onUseWorkspaceFile?.();
              }}
            >
              <Icon icon={FolderOpenIcon} className="mr-2 h-4 w-4" />
              Use File from Workspace
              {isAtCap && <CapLabel />}
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
          {extraItems.length > 0 && <DropdownMenuSeparator />}
          {extraItems.map((item) => (
            <DropdownMenuItem
              key={item.key}
              disabled={item.disabled}
              onSelect={item.onSelect}
            >
              <Icon icon={item.icon} className="mr-2 h-4 w-4" />
              {item.label}
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>
    </>
  );
}

function CapLabel() {
  return (
    <span className="ml-auto pl-3 text-xs text-foreground-tertiary">
      {MAX_ATTACHMENTS} max
    </span>
  );
}
