"use client";

import { cn } from "@/lib/utils";

type Props = {
  children: React.ReactNode;
  className?: string;
};

export function SectionWrap({ children, className }: Props) {
  return (
    <div
      className={cn(
        // TODO(tokens): no exact role for bg-[#FAFAFA]
        "flex min-h-0 flex-col gap-4 rounded-medium bg-[#FAFAFA] py-4",
        className,
      )}
    >
      {children}
    </div>
  );
}
