import type { ReactNode } from "react";

interface Props {
  children: ReactNode;
}

export function PanelTray({ children }: Props) {
  return (
    <div className="flex h-full min-h-0 w-full flex-col rounded-3xl bg-surface-tray p-1.5">
      <div className="min-h-0 flex-1 overflow-hidden rounded-2xl border border-border-card bg-surface shadow-sm">
        {children}
      </div>
    </div>
  );
}
