"use client";

import { Button } from "@/components/atoms/Button/Button";
import { ExpertReviewDialog } from "@/components/contextual/ExpertReviewDialog/ExpertReviewDialog";
import { Upload03Icon } from "@hugeicons/core-free-icons";
import { useImportExpertButton } from "./useImportExpertButton";

interface Props {
  /** `link` is the empty-state's inline phrasing; `button` is the header's. */
  variant?: "button" | "link";
}

export function ImportExpertButton({ variant = "button" }: Props) {
  const {
    fileInputRef,
    isParsing,
    isImporting,
    isDialogOpen,
    preview,
    openFilePicker,
    handleFileChange,
    closeDialog,
    confirmImport,
  } = useImportExpertButton();

  return (
    <>
      <input
        ref={fileInputRef}
        type="file"
        accept=".zip,application/zip"
        className="hidden"
        onChange={handleFileChange}
        data-testid={`expert-import-input-${variant}`}
      />
      <Button
        variant={variant === "link" ? "link" : "secondary"}
        size="small"
        leadingIcon={variant === "link" ? undefined : Upload03Icon}
        loading={isParsing}
        onClick={openFilePicker}
        data-testid={`expert-import-${variant}`}
      >
        {variant === "link" ? "Import from a file" : "Import expert"}
      </Button>

      <ExpertReviewDialog
        mode="import"
        open={isDialogOpen}
        preview={preview}
        isSubmitting={isImporting}
        onClose={closeDialog}
        onConfirm={confirmImport}
      />
    </>
  );
}
