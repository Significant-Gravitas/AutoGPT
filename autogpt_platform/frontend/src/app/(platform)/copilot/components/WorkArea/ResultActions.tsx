"use client";

import type { Task } from "@/app/api/__generated__/models/task";
import type { TaskArtifact } from "@/app/api/__generated__/models/taskArtifact";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { FOCUS } from "../EmptySession/components/ExpertInfoPanel/components/InfoParts";
import { fileSourceURL } from "../TaskHandoff/helpers";
import { fileNameOf, resultMarkdown, saveFile, saveText } from "./resultFiles";
import { useFlash } from "./useFlash";

interface Props {
  task: Task;
  /** The result in words, or what went wrong. */
  text: string | null;
  /** The document holding the full result, when there is one. */
  document: TaskArtifact | null;
}

/** Take the result away: Copy its words, or Download it (the result's own
 *  document when it has one, else a Markdown file named after the task). */
export function ResultActions({ task, text, document }: Props) {
  const { flashed, flash } = useFlash();

  async function copy() {
    if (!text) return;
    await navigator.clipboard.writeText(text);
    flash("copied");
  }

  function download() {
    if (document?.workspace_file_id) {
      saveFile(fileSourceURL(document.workspace_file_id), document.title);
    } else if (text) {
      saveText(
        resultMarkdown(task, text),
        fileNameOf(task.title, "md"),
        "text/markdown",
      );
    } else return;
    flash("saved");
  }

  return (
    <span className="flex items-center gap-3">
      {text ? (
        <HeaderAction onClick={() => void copy()}>
          {flashed === "copied" ? "Copied" : "Copy"}
        </HeaderAction>
      ) : null}
      {text || document ? (
        <HeaderAction onClick={download}>
          {flashed === "saved" ? "Saved" : "Download"}
        </HeaderAction>
      ) : null}
    </span>
  );
}

interface ActionProps {
  onClick: () => void;
  children: string;
}

/** A quiet word at the end of a block's label row, like "View all". */
function HeaderAction({ onClick, children }: ActionProps) {
  return (
    <button
      type="button"
      aria-live="polite"
      onClick={onClick}
      className={cn("rounded-md hover:underline", FOCUS)}
    >
      <Text variant="small" as="span" tone="quiet">
        {children}
      </Text>
    </button>
  );
}
