"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import CodeMirror from "@uiw/react-codemirror";
import { type KeyboardEvent, useState } from "react";
import { useCopilotUIStore } from "../../../../store";
import {
  codeHighlighting,
  editorTheme,
  getLanguageExtension,
} from "../../helpers";
import { LineCommentPopover } from "./LineCommentPopover";
import {
  type LineCommentRequest,
  lineCommentButton,
} from "./lineCommentButton";
import { useFileEditor } from "./useFileEditor";

interface Props {
  sessionId: string;
  path: string;
}

export function FileEditor({ sessionId, path }: Props) {
  const { value, setValue, save, isLoading, isError, truncated } =
    useFileEditor(sessionId, path);
  const insertIntoChatInput = useCopilotUIStore((s) => s.insertIntoChatInput);
  const addCodeRef = useCopilotUIStore((s) => s.addCodeRef);
  const [commentRequest, setCommentRequest] =
    useState<LineCommentRequest | null>(null);

  function handleAddCodeRef(instruction: string) {
    if (!commentRequest) return;
    addCodeRef({
      id: crypto.randomUUID(),
      path,
      fromLine: commentRequest.fromLine,
      toLine: commentRequest.toLine,
      code: commentRequest.code,
    });
    if (instruction) insertIntoChatInput(instruction);
    setCommentRequest(null);
  }

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s") {
      event.preventDefault();
      save();
    }
  }

  if (isLoading) {
    return (
      <div className="p-3">
        <Skeleton className="h-full w-full" />
      </div>
    );
  }

  if (isError) {
    return (
      <div className="p-3 text-sm text-zinc-400">Couldn’t load this file.</div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col" onKeyDown={handleKeyDown}>
      {truncated ? (
        <div className="shrink-0 bg-amber-50 px-3 py-1.5 text-xs text-amber-700">
          Large file — showing the first 1 MB (read-only).
        </div>
      ) : null}
      <div className="min-h-0 flex-1 overflow-auto">
        <CodeMirror
          value={value}
          onChange={setValue}
          editable={!truncated}
          readOnly={truncated}
          basicSetup={{ foldGutter: false, highlightActiveLine: !truncated }}
          extensions={[
            editorTheme,
            codeHighlighting,
            lineCommentButton({ onRequestComment: setCommentRequest }),
            ...getLanguageExtension(path),
          ]}
          className="h-full p-2 text-[13px]"
        />
      </div>
      {commentRequest ? (
        <LineCommentPopover
          request={commentRequest}
          onSubmit={handleAddCodeRef}
          onClose={() => setCommentRequest(null)}
        />
      ) : null}
    </div>
  );
}
