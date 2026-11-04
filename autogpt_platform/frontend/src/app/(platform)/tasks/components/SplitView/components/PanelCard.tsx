import type { ReactNode } from "react";

interface Props {
  children: ReactNode;
}

// The padding keeps the card's shadow edge inside the clipping panel.
export function PanelCard({ children }: Props) {
  return (
    <div className="flex h-full min-h-0 w-full flex-col p-0.5">
      <div className="min-h-0 flex-1 overflow-hidden rounded-card bg-surface shadow-card">
        {children}
      </div>
    </div>
  );
}
