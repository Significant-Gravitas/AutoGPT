import type { ReactNode } from "react";

interface Props {
  children: ReactNode;
}

export function TasksCard({ children }: Props) {
  return (
    <div className="overflow-hidden rounded-card bg-surface shadow-(--inbox-shadow-card)">
      {children}
    </div>
  );
}
