"use client";

import { ChatCircleIcon } from "@/components/atoms/AGPTIcon/icons";
import { Button } from "@/components/atoms/Button/Button";
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { LineCommentRequest } from "./lineCommentButton";

interface Props {
  request: LineCommentRequest;
  onSubmit: (instruction: string) => void;
  onClose: () => void;
}

const POPOVER_WIDTH = 288;

export function LineCommentPopover({ request, onSubmit, onClose }: Props) {
  const [value, setValue] = useState("");
  const containerRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(function focusOnOpen() {
    textareaRef.current?.focus();
  }, []);

  useEffect(
    function dismissHandlers() {
      function handleKey(event: KeyboardEvent) {
        if (event.key === "Escape") onClose();
      }
      function handleOutside(event: MouseEvent) {
        if (
          containerRef.current &&
          !containerRef.current.contains(event.target as Node)
        ) {
          onClose();
        }
      }
      document.addEventListener("keydown", handleKey);
      document.addEventListener("mousedown", handleOutside);
      return () => {
        document.removeEventListener("keydown", handleKey);
        document.removeEventListener("mousedown", handleOutside);
      };
    },
    [onClose],
  );

  const left = Math.min(
    request.anchorRect.right + 8,
    window.innerWidth - POPOVER_WIDTH - 8,
  );
  const top = Math.min(request.anchorRect.top, window.innerHeight - 240);

  function handleSubmit() {
    onSubmit(value.trim());
  }

  return createPortal(
    <div
      ref={containerRef}
      style={{ position: "fixed", top, left, width: POPOVER_WIDTH }}
      className="z-50 rounded-3xl border border-zinc-200 bg-white p-2 shadow-lg [corner-shape:squircle]"
    >
      <span className="mb-1.5 flex items-center gap-1.5 px-1 text-sm font-medium text-zinc-800">
        <ChatCircleIcon size={16} />
        Temporary comment
      </span>
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            handleSubmit();
          }
        }}
        rows={2}
        placeholder="Describe the change you want…"
        className="w-full resize-none rounded-lg border-none px-2 py-1.5 text-sm text-zinc-800 placeholder:text-zinc-400 focus:outline-none"
      />
      <div className="mt-2 flex justify-end gap-1.5">
        <Button variant="ghost" size="small" onClick={onClose}>
          Cancel
        </Button>
        <Button variant="primary" size="small" onClick={handleSubmit}>
          Add to chat
        </Button>
      </div>
    </div>,
    document.body,
  );
}
