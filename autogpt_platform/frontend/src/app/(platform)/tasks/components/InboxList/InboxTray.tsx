import type { ReactNode } from "react";

interface Props {
  children: ReactNode;
}

export function InboxTray({ children }: Props) {
  return (
    <div className="rounded-3xl bg-surface-tray p-1.5">
      <div className="overflow-hidden rounded-2xl border border-border-card bg-surface shadow-sm">
        {children}
      </div>
    </div>
  );
}
