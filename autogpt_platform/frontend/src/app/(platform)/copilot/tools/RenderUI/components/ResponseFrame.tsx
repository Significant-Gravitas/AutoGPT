import type { ReactNode } from "react";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Layout01Icon } from "@hugeicons/core-free-icons";
import { cn } from "@/lib/utils";

interface Props {
  children: ReactNode;
  valid: boolean;
  view: "interactive" | "summary";
  setView: (view: "interactive" | "summary") => void;
  locked: boolean;
  isSending: boolean;
  sent: boolean;
  error: string | null;
}

export function ResponseFrame(props: Props) {
  const { valid, view, setView, children, error } = props;
  return (
    <section
      aria-label="Interactive response"
      className="my-3 min-w-0 overflow-hidden rounded-xl border border-zinc-200 bg-zinc-50"
    >
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-zinc-200 bg-white px-4 py-3">
        <span className="flex items-center gap-2 text-xs font-medium text-zinc-600">
          <Icon icon={Layout01Icon} size={16} />
          Interactive view
        </span>
        {valid && (
          <div className="flex gap-1" aria-label="Response presentation">
            {(["interactive", "summary"] as const).map((value) => (
              <button
                key={value}
                onClick={() => setView(value)}
                aria-pressed={view === value}
                className={cn(
                  "rounded-md px-3 py-1.5 text-xs text-zinc-600 focus-visible:outline-purple-500",
                  view === value && "bg-zinc-100 text-zinc-900",
                )}
              >
                {value === "interactive" ? "Explore" : "Summary"}
              </button>
            ))}
          </div>
        )}
      </div>
      {children}
      {error && (
        <p
          role="alert"
          className="border-t border-red-100 bg-red-50 px-4 py-3 text-xs text-red-700"
        >
          {error}
        </p>
      )}
      <div
        className="border-t border-zinc-200 bg-white px-4 py-2 text-[11px] text-zinc-500"
        role="status"
      >
        {statusText(props)}
      </div>
    </section>
  );
}

function statusText({ locked, isSending, sent }: Props) {
  if (locked) return "Shared view · conversation actions are disabled";
  if (isSending) return "Sending to the conversation…";
  if (sent) return "Sent to the conversation";
  return "Changes to inputs stay in this tab. Submit to continue the conversation.";
}
