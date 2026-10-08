"use client";

import type { RenderUIMessagePart } from "./isRenderUIPart";
import { OpenUI } from "@/components/organisms/OpenUI/OpenUI";
import { SavedSummary } from "./components/SavedSummary";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Loading03Icon } from "@hugeicons/core-free-icons";
import { ResponseFrame } from "./components/ResponseFrame";
import { getStreamingSource, readUIResult } from "./helpers";
import { useRenderUI } from "./useRenderUI";

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
  const { result, valid, title, summary } = readUIResult(part);
  const pending =
    part.state === "input-streaming" || part.state === "input-available";
  const source = getStreamingSource(part);
  if (pending && isCurrentlyStreaming) {
    return (
      <section
        aria-label="Interactive view being created"
        className="my-3 min-w-0 space-y-4 rounded-xl border border-zinc-200 bg-zinc-50 p-5"
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
      key={part.toolCallId}
      draftID={
        result?.session_id ? `${result.session_id}:${part.toolCallId}` : null
      }
      source={result?.source ?? ""}
      title={title}
      summary={summary}
      valid={valid}
      readOnly={readOnly}
    />
  );
}

interface CompletedProps {
  draftID: string | null;
  source: string;
  title: string;
  summary: string;
  valid: boolean;
  readOnly: boolean;
}

function CompletedUI({
  draftID,
  source,
  title,
  summary,
  valid,
  readOnly,
}: CompletedProps) {
  const ui = useRenderUI(draftID, source, title, readOnly);
  return (
    <ResponseFrame valid={valid} {...ui}>
      <div className="max-h-[44rem] overflow-auto p-4 sm:p-5">
        {valid && (
          <div hidden={ui.view !== "interactive"}>
            <OpenUI
              source={source}
              isStreaming={false}
              disabled={ui.locked || ui.isSending}
              revision={1}
              initialState={ui.initialState}
              onStateUpdate={ui.onStateUpdate}
              onAction={ui.onAction}
              fallback={
                <p className="whitespace-pre-wrap text-sm">{summary}</p>
              }
            />
          </div>
        )}
        {(!valid || ui.view === "summary") && (
          <SavedSummary
            summary={summary}
            valid={valid}
            locked={ui.locked}
            isSending={ui.isSending}
            onRebuild={() =>
              void ui.send(
                "Please rebuild the previous interactive view using the same data and valid OpenUI components.",
              )
            }
          />
        )}
      </div>
    </ResponseFrame>
  );
}
