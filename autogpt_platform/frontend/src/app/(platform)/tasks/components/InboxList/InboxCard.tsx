import type { ReactNode } from "react";

interface Props {
  children: ReactNode;
}

export function InboxCard({ children }: Props) {
  return (
    <div className="overflow-hidden rounded-card bg-surface shadow-card">
      {children}
    </div>
  );
}
