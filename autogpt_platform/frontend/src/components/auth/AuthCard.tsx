import { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface Props {
  children: ReactNode;
  className?: string;
}

export default function AuthCard({ children, className }: Props) {
  return (
    <div
      className={cn(
        "flex h-[80vh] w-[32rem] items-center justify-center",
        className,
      )}
    >
      <div className="w-full max-w-md rounded-lg bg-white p-6 shadow-md">
        {children}
      </div>
    </div>
  );
}
