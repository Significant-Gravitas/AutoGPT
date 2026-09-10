import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { ChangeEvent, useRef } from "react";
import { useCopilotModal } from "../../../useCopilotModal";

interface Args {
  onFilesSelected: (files: File[]) => void;
}

export function useComposerPlusMenu({ onFilesSelected }: Args) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { openModal } = useCopilotModal();
  const showWorkspaceOption = useGetFlag(Flag.CHAT_WORKSPACE_FILES);

  function openFilePicker() {
    fileInputRef.current?.click();
  }

  function handleFileChange(e: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    if (files.length > 0) {
      onFilesSelected(files);
    }
    // Reset so the same file can be re-selected
    e.target.value = "";
  }

  return {
    fileInputRef,
    openModal,
    showWorkspaceOption,
    openFilePicker,
    handleFileChange,
  };
}
