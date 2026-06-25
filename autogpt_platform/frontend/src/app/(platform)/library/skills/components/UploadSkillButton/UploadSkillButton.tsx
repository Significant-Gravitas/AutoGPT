"use client";

import { Button } from "@/components/atoms/Button/Button";
import { UploadSimpleIcon } from "@phosphor-icons/react";
import { useUploadSkillButton } from "./useUploadSkillButton";

export function UploadSkillButton() {
  const { fileInputRef, isUploading, openFilePicker, handleFileChange } =
    useUploadSkillButton();

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
      <Button
        variant="primary"
        size="small"
        onClick={openFilePicker}
        loading={isUploading}
        data-testid="skill-upload-button"
      >
        <UploadSimpleIcon className="mr-1 h-4 w-4" />
        Upload skill
      </Button>
    </>
  );
}
