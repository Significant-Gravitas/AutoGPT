"use client";

import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";
import { Plus as PlusIcon } from "@phosphor-icons/react";
import { useRef } from "react";

interface Props {
  onFilesSelected: (files: File[]) => void;
  disabled?: boolean;
}

export function AttachmentMenu({ onFilesSelected, disabled }: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  function handleClick() {
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
      <Button
        type="button"
        variant="icon"
        size="icon"
        aria-label="Attach file"
        disabled={disabled}
        onClick={handleClick}
        className={cn(
          "border-zinc-300 bg-white text-zinc-500 hover:border-zinc-400 hover:bg-zinc-50 hover:text-zinc-700",
          disabled && "opacity-40",
        )}
      >
        <PlusIcon className="h-4 w-4" weight="bold" />
      </Button>
    </>
  );
}
