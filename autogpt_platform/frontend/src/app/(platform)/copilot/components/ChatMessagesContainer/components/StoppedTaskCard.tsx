import { Lightbulb, Square } from "@phosphor-icons/react";

export function StoppedTaskCard() {
  return (
    <div className="rounded-medium my-2 flex items-start gap-3 border border-zinc-200 bg-zinc-50 p-4">
      <div className="rounded-small flex h-9 w-9 shrink-0 items-center justify-center bg-purple-50">
        <Square size={16} weight="fill" className="text-purple-500" />
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-sm font-semibold text-zinc-900">Task stopped</p>
        <p className="mt-1 text-[13px] text-zinc-600">
          The response above is incomplete. You can ask to continue or type
          something new.
        </p>
        <div className="mt-2.5 flex items-center gap-1.5 text-xs text-slate-400">
          <Lightbulb size={14} className="shrink-0 text-purple-300" />
          Try &ldquo;continue&rdquo; or type something new.
        </div>
      </div>
    </div>
  );
}
