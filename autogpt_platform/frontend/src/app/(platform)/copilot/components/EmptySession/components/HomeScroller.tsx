import type { ReactNode } from "react";

interface Props {
  children: ReactNode;
}

/** Home's scrolling column. */
export function HomeScroller({ children }: Props) {
  return (
    <div className="relative scrollbar-none flex min-h-0 flex-1 items-start justify-center overflow-y-auto px-0 py-5 md:px-6 md:py-10">
      {children}
    </div>
  );
}
