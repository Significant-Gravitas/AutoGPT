"use client";

import { Button } from "@/components/atoms/Button/Button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { UploadSimpleIcon } from "@phosphor-icons/react";
import { useUploadSkillButton } from "./useUploadSkillButton";

interface Props {
  onUploaded?: (name: string) => void;
}

export function UploadSkillButton({ onUploaded }: Props) {
  const { fileInputRef, isUploading, openFilePicker, handleFileChange } =
    useUploadSkillButton({ onUploaded });

  return (
    <>
      <input
        ref={fileInputRef}
        type="file"
        accept=".md,.markdown,text/markdown"
        className="hidden"
        onChange={handleFileChange}
        data-testid="skill-upload-input"
      />
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="secondary"
            size="small"
            onClick={openFilePicker}
            loading={isUploading}
            data-testid="skill-upload-button"
          >
            <UploadSimpleIcon className="mr-1 h-4 w-4" />
            Upload skill
          </Button>
        </TooltipTrigger>
        <TooltipContent side="bottom">
          Import a skill file you&apos;ve exported
        </TooltipContent>
      </Tooltip>
    </>
  );
}
