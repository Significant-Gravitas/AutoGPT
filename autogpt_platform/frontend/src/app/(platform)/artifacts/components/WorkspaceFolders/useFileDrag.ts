import { useEffect, useRef } from "react";
import type { DragEvent } from "react";
import { createFileDragImage, writeFileDragData } from "./drag";

/**
 * Drag handlers for a file card/row so it can be dropped onto a folder. Owns
 * the off-screen drag-image node and removes it on dragend or unmount.
 * `fileIds` are the files carried by the drag; `label` is shown in the chip
 * under the cursor.
 */
export function useFileDrag(fileIds: string[], label: string) {
  const dragImageRef = useRef<HTMLElement | null>(null);

  // Clean up a leftover drag-image node if the element unmounts mid-drag
  // (before dragend fires), so the off-screen node isn't leaked into the DOM.
  useEffect(() => {
    return () => {
      dragImageRef.current?.remove();
      dragImageRef.current = null;
    };
  }, []);

  function handleDragStart(e: DragEvent<HTMLElement>) {
    dragImageRef.current?.remove();
    writeFileDragData(e.dataTransfer, fileIds);
    e.dataTransfer.effectAllowed = "move";
    const dragImage = createFileDragImage(label);
    document.body.appendChild(dragImage);
    e.dataTransfer.setDragImage(dragImage, 16, 16);
    dragImageRef.current = dragImage;
  }

  function handleDragEnd() {
    dragImageRef.current?.remove();
    dragImageRef.current = null;
  }

  return { handleDragStart, handleDragEnd };
}
