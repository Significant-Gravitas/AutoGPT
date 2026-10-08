import { useEffect, useRef, useState } from "react";
import type { ActionEvent } from "@openuidev/react-lang";
import { OpenUI } from "@/components/organisms/OpenUI/OpenUI";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  Download01Icon,
  ReplayIcon,
  Loading03Icon,
} from "@hugeicons/core-free-icons";
import { cn } from "@/lib/utils";
import { workspaceText } from "@/lib/openui/parse";
import { exportWorkspace } from "../../helpers";

interface Props {
  source: string;
  isStreaming: boolean;
  revision: number;
  mode: "sample" | "live";
  onAction: (event: ActionEvent) => void;
  onReplay: () => void;
}
const views = ["Interactive", "Text response", "Source"] as const;

export function WorkspacePanel({
  source,
  isStreaming,
  revision,
  mode,
  onAction,
  onReplay,
}: Props) {
  const [view, setView] = useState<(typeof views)[number]>("Interactive");
  const content = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (content.current) content.current.scrollTop = 0;
  }, [revision, view]);
  return (
    <section
      className="flex h-full min-h-0 min-w-0 flex-col overflow-hidden rounded-2xl border border-zinc-200 bg-zinc-50"
      aria-label="Generated workspace"
    >
      <div className="flex min-h-14 shrink-0 flex-wrap items-center justify-between gap-2 border-b border-zinc-200 bg-white px-3 py-2 sm:px-5">
        <div className="flex gap-1" aria-label="Workspace view">
          {views.map((value) => (
            <button
              key={value}
              aria-pressed={view === value}
              onClick={() => setView(value)}
              className={cn(
                "rounded-md px-2.5 py-1.5 text-[11px] font-medium text-zinc-500 transition-colors focus-visible:outline-purple-500",
                view === value && "bg-zinc-100 text-zinc-900",
              )}
            >
              {value}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-1">
          {mode === "sample" && (
            <Button
              variant="ghost"
              size="icon-xs"
              aria-label="Replay sample"
              disabled={isStreaming}
              onClick={onReplay}
            >
              <Icon icon={ReplayIcon} size={15} />
            </Button>
          )}
          <Button
            variant="ghost"
            size="xs"
            aria-label="Export UI"
            disabled={isStreaming || !source}
            onClick={() => exportWorkspace(source)}
            leadingIcon={Download01Icon}
          >
            Export
          </Button>
        </div>
      </div>
      <div
        className="min-h-0 flex-1 overflow-y-auto p-4 sm:p-6"
        ref={content}
        aria-busy={isStreaming}
      >
        {!source && isStreaming ? (
          <div className="flex min-h-64 items-center justify-center gap-2 text-sm text-zinc-500">
            <Icon icon={Loading03Icon} size={18} className="animate-spin" />
            Building your workspace…
          </div>
        ) : null}
        <div hidden={view !== "Interactive"}>
          <OpenUI
            source={source}
            isStreaming={isStreaming}
            onAction={onAction}
            revision={revision}
          />
        </div>
        {view === "Source" ? (
          <div>
            <p className="mb-4 text-xs text-zinc-500">
              OpenUI Lang · composed from AutoGPT components
            </p>
            <pre
              aria-label="OpenUI source"
              className="whitespace-pre-wrap break-words rounded-xl border border-zinc-200 bg-white p-5 font-mono text-xs leading-6 text-zinc-700"
            >
              {source}
            </pre>
          </div>
        ) : view === "Text response" ? (
          <div className="rounded-xl border border-zinc-200 bg-white p-6">
            <p className="mb-5 text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
              The same response, as text
            </p>
            <div className="whitespace-pre-wrap text-sm leading-relaxed text-zinc-700">
              {workspaceText(source)}
            </div>
          </div>
        ) : null}
      </div>
      <div className="flex shrink-0 items-center justify-between gap-3 border-t border-zinc-200 bg-white px-5 py-3 text-[10px] text-zinc-400">
        <span className="flex items-center gap-1.5">
          <span
            className={cn(
              "size-1.5 rounded-full bg-green-400",
              isStreaming && "bg-purple-400 motion-safe:animate-pulse",
            )}
          />
          {isStreaming ? "Streaming response" : "Ready to explore"}
        </span>
        <span>
          {mode === "sample"
            ? "Sample data · no connected accounts"
            : "AI-generated · review before use"}
        </span>
      </div>
    </section>
  );
}
