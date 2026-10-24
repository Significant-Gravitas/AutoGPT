import { cn } from "@/lib/utils";
import { ReactNode } from "react";

interface Props {
  children: ReactNode;
  className?: string;
}

export function Card({ children, className }: Props) {
  return (
    <div
      data-slot="card"
      className={cn("rounded-large bg-surface p-6 shadow-md", className)}
    >
      {children}
    </div>
  );
}
