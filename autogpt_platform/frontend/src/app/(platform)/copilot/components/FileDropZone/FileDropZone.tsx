"use client";

import { cn } from "@/lib/utils";
import { UploadSimple } from "@phosphor-icons/react";
import { useRef, useState } from "react";

interface Props {
  children: React.ReactNode;
  onFilesDropped: (files: File[]) => void;
  className?: string;
}

/**
 * Wraps children with drag-and-drop file handling. Owns its own drag state so
 * that the overlay can toggle without re-rendering the rest of the tree.
 */
export function FileDropZone({ children, onFilesDropped, className }: Props) {
  const [isDragging, setIsDragging] = useState(false);
  const dragCounter = useRef(0);

  function handleDragEnter(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current += 1;
    if (e.dataTransfer.types.includes("Files")) {
      setIsDragging(true);
    }
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
  }

  function handleDragLeave(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current -= 1;
    if (dragCounter.current === 0) {
      setIsDragging(false);
    }
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current = 0;
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      onFilesDropped(files);
    }
  }

  return (
    <div
      className={className}
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {children}
      <div
        className={cn(
          "pointer-events-none absolute inset-0 z-50 flex flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed border-violet-400 bg-violet-500/10 transition-opacity duration-150",
          isDragging ? "opacity-100" : "opacity-0",
        )}
      >
        <UploadSimple className="h-10 w-10 text-violet-500" weight="bold" />
        <span className="text-lg font-medium text-violet-600">
          Drop files here
        </span>
      </div>
    </div>
  );
}
