"use client";

import { MergeView } from "@codemirror/merge";
import { EditorState, EditorView } from "@uiw/react-codemirror";
import { useEffect, useRef } from "react";
import { editorTheme, getLanguageExtension } from "../../helpers";

interface Props {
  path: string;
  original: string;
  modified: string;
}

export function DiffView({ path, original, modified }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const parent = containerRef.current;
    if (!parent) return;

    const readOnly = [
      EditorView.editable.of(false),
      EditorState.readOnly.of(true),
      editorTheme,
      EditorView.lineWrapping,
      ...getLanguageExtension(path),
    ];

    const view = new MergeView({
      a: { doc: original, extensions: readOnly },
      b: { doc: modified, extensions: readOnly },
      parent,
    });

    return () => view.destroy();
  }, [path, original, modified]);

  return (
    <div ref={containerRef} className="h-full overflow-auto text-[13px]" />
  );
}
