"use client";

import {
  SplitHorizontalIcon,
  SplitVerticalIcon,
  XIcon,
} from "@phosphor-icons/react";
import { useSplitPaneContext } from "./SplitPaneContext";

interface Props {
  paneId: string;
  title: string | null;
}

export function PaneToolbar({ paneId, title }: Props) {
  const { splitPane, closePane, leafCount, focusedPaneId } =
    useSplitPaneContext();

  const isFocused = focusedPaneId === paneId;
  const canClose = leafCount > 1;

  return (
    <div
      className={
        "flex h-8 shrink-0 items-center justify-between border-b px-2 text-xs " +
        (isFocused
          ? "border-violet-200 bg-violet-50 text-violet-700"
          : "border-zinc-200 bg-zinc-50 text-zinc-500")
      }
    >
      <span className="min-w-0 truncate font-medium">
        {title || "New chat"}
      </span>
      <div className="flex items-center gap-0.5">
        <button
          onClick={() => splitPane(paneId, "horizontal")}
          className="rounded p-1 hover:bg-black/5"
          aria-label="Split horizontally"
          title="Split horizontal (side by side)"
        >
          <SplitHorizontalIcon className="h-3.5 w-3.5" />
        </button>
        <button
          onClick={() => splitPane(paneId, "vertical")}
          className="rounded p-1 hover:bg-black/5"
          aria-label="Split vertically"
          title="Split vertical (top/bottom)"
        >
          <SplitVerticalIcon className="h-3.5 w-3.5" />
        </button>
        {canClose && (
          <button
            onClick={() => closePane(paneId)}
            className="rounded p-1 hover:bg-red-100 hover:text-red-600"
            aria-label="Close pane"
            title="Close pane"
          >
            <XIcon className="h-3.5 w-3.5" />
          </button>
        )}
      </div>
    </div>
  );
}
