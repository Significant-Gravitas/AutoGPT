"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import CodeMirror, { EditorView } from "@uiw/react-codemirror";
import { type KeyboardEvent } from "react";
import { editorTheme, getLanguageExtension } from "../../helpers";
import { useFileEditor } from "./useFileEditor";

interface Props {
  sessionId: string;
  path: string;
}

export function FileEditor({ sessionId, path }: Props) {
  const { value, setValue, save, isLoading, isError, truncated } =
    useFileEditor(sessionId, path);

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
          theme="light"
          basicSetup={{ foldGutter: false, highlightActiveLine: !truncated }}
          extensions={[
            editorTheme,
            EditorView.lineWrapping,
            ...getLanguageExtension(path),
          ]}
          className="h-full text-[13px]"
        />
      </div>
    </div>
  );
}
