import { cn } from "@/lib/utils";
import { ReactNode } from "react";

interface Props {
  children: ReactNode;
  className?: string;
}

export function Card({ children, className }: Props) {
  return (
    <div className={cn("rounded-large bg-white p-6 shadow-md", className)}>
      {children}
    </div>
  );
}
