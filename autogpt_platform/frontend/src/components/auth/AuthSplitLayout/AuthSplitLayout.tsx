import { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface Props {
  marketing: ReactNode;
  children: ReactNode;
  className?: string;
}

export function AuthSplitLayout({ marketing, children, className }: Props) {
  return (
    <div className={cn("flex min-h-screen w-full flex-1", className)}>
      <aside className="relative hidden flex-1 overflow-hidden bg-slate-950 text-white lg:flex">
        {marketing}
      </aside>
      <section className="flex flex-1 flex-col items-center justify-center bg-white px-6 py-12 sm:px-10">
        <div className="flex w-full max-w-[26rem] flex-col">{children}</div>
      </section>
    </div>
  );
}
