"use client";

import { Button } from "@/components/atoms/Button/Button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { cn } from "@/lib/utils";
import {
  FileText as FileTextIcon,
  Image as ImageIcon,
  MusicNote as MusicNoteIcon,
  Plus as PlusIcon,
  Table as TableIcon,
  VideoCamera as VideoCameraIcon,
} from "@phosphor-icons/react";
import { useRef, useState } from "react";

const FILE_CATEGORIES = [
  {
    label: "Documents",
    icon: FileTextIcon,
    accept:
      ".pdf,.doc,.docx,.txt,.rtf,.odt,.md,.json,.xml,.html,.htm,.yaml,.yml,.toml",
  },
  {
    label: "Spreadsheets",
    icon: TableIcon,
    accept: ".xlsx,.xls,.csv,.tsv,.ods",
  },
  {
    label: "Images",
    icon: ImageIcon,
    accept: ".png,.jpg,.jpeg,.gif,.webp,.svg,.bmp,.ico,.tiff",
  },
  {
    label: "Video",
    icon: VideoCameraIcon,
    accept: ".mp4,.webm,.mov,.avi,.mkv",
  },
  {
    label: "Audio",
    icon: MusicNoteIcon,
    accept: ".mp3,.wav,.ogg,.flac,.aac,.m4a,.wma",
  },
] as const;

/** Set of all allowed file extensions (e.g. ".pdf", ".png") derived from FILE_CATEGORIES. */
export const ALLOWED_EXTENSIONS: ReadonlySet<string> = new Set(
  FILE_CATEGORIES.flatMap((c) => c.accept.split(",")),
);

interface Props {
  onFilesSelected: (files: File[]) => void;
  disabled?: boolean;
}

export function AttachmentMenu({ onFilesSelected, disabled }: Props) {
  const [open, setOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const acceptRef = useRef("");

  function handleCategoryClick(accept: string) {
    acceptRef.current = accept;
    if (fileInputRef.current) {
      fileInputRef.current.accept = accept;
      fileInputRef.current.click();
    }
    setOpen(false);
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
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            type="button"
            variant="icon"
            size="icon"
            aria-label="Attach file"
            disabled={disabled}
            className={cn(
              "border-zinc-300 bg-white text-zinc-500 hover:border-zinc-400 hover:bg-zinc-50 hover:text-zinc-700",
              disabled && "opacity-40",
            )}
          >
            <PlusIcon className="h-4 w-4" weight="bold" />
          </Button>
        </PopoverTrigger>
        <PopoverContent
          side="top"
          align="start"
          sideOffset={8}
          className="w-48 p-1"
        >
          {FILE_CATEGORIES.map((category) => (
            <button
              key={category.label}
              type="button"
              onClick={() => handleCategoryClick(category.accept)}
              className="flex w-full items-center gap-3 rounded-md px-3 py-2 text-sm text-zinc-700 transition-colors hover:bg-zinc-100"
            >
              <category.icon className="h-4 w-4 text-zinc-500" />
              {category.label}
            </button>
          ))}
        </PopoverContent>
      </Popover>
    </>
  );
}
