"use client";

import type { RenderUIMessagePart } from "./isRenderUIPart";
import { OpenUI } from "@/components/organisms/OpenUI/OpenUI";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Loading03Icon } from "@hugeicons/core-free-icons";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import {
  getStreamingSource,
  readSavedFile,
  readUIError,
  readUIResult,
  type SavedFileRef,
} from "./helpers";
import { useRenderUI } from "./useRenderUI";
import { RejectedUI } from "./components/RejectedUI";
import { ChatViewCard } from "./components/ChatViewCard/ChatViewCard";
import { SavedViewCard } from "./components/SavedViewCard";

interface Props {
  part: RenderUIMessagePart;
  readOnly?: boolean;
  isCurrentlyStreaming?: boolean;
}

export function RenderUI({
  part,
  readOnly = false,
  isCurrentlyStreaming = false,
}: Props) {
  const ownerId = useAuthStore((state) => state.user?.id);
  const error = readUIError(part);
  if (error)
    return <RejectedUI part={part} message={error} readOnly={readOnly} />;
  const { result, valid, title, summary } = readUIResult(part);
  const pending =
    part.state === "input-streaming" || part.state === "input-available";
  const source = getStreamingSource(part);
  if (pending && isCurrentlyStreaming) {
    return (
      <section
        aria-label="Interactive view being created"
        className="my-3 min-w-0 space-y-4"
        aria-busy
      >
        <div className="flex items-center gap-2 text-xs text-zinc-500">
          <Icon icon={Loading03Icon} size={16} className="animate-spin" />
          Creating an interactive view…
        </div>
        {source && (
          <OpenUI
            source={source}
            isStreaming
            disabled
            revision={0}
            onAction={() => {}}
          />
        )}
      </section>
    );
  }
  return (
    <CompletedUI
      key={`${ownerId ?? "anonymous"}:${part.toolCallId}:${result?.source ?? ""}`}
      sessionId={result?.session_id ?? null}
      toolCallId={part.toolCallId}
      file={readSavedFile(result)}
      source={result?.source ?? ""}
      title={title}
      summary={summary}
      valid={valid}
      readOnly={readOnly}
    />
  );
}

interface CompletedProps {
  sessionId: string | null;
  toolCallId: string;
  file: SavedFileRef | null;
  source: string;
  title: string;
  summary: string;
  valid: boolean;
  readOnly: boolean;
}

function CompletedUI({
  sessionId,
  toolCallId,
  file,
  source,
  title,
  summary,
  valid,
  readOnly,
}: CompletedProps) {
  const ui = useRenderUI({ sessionId, toolCallId, source, file, readOnly });
  if (valid && ui.savedFile)
    return (
      <SavedViewCard
        file={ui.savedFile}
        sessionId={sessionId}
        source={source}
        title={title}
      />
    );
  return (
    <ChatViewCard
      ownerId={ui.ownerId}
      sessionId={sessionId ?? ""}
      toolCallId={toolCallId}
      viewId={ui.viewId}
      stable={ui.stable}
      locked={ui.locked}
      remote={ui.remote}
      source={source}
      title={title}
      summary={summary}
      valid={valid}
      onSaved={ui.onSaved}
    />
  );
}
