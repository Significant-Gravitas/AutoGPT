export function TypingIndicator() {
  return (
    <div
      role="status"
      aria-live="polite"
      aria-label="Assistant is typing"
      className="flex max-w-[85%] items-center gap-1 rounded-lg bg-slate-100 px-3 py-3"
    >
      <span
        aria-hidden="true"
        className="h-2 w-2 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.3s]"
      />
      <span
        aria-hidden="true"
        className="h-2 w-2 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.15s]"
      />
      <span
        aria-hidden="true"
        className="h-2 w-2 animate-bounce rounded-full bg-slate-400"
      />
    </div>
  );
}
