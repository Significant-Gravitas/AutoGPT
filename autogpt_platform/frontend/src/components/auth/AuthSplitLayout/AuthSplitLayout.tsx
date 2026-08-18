import { ReactNode } from "react";
import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { AuroraBackground } from "@/components/ui/aurora-background";
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
      <section className="relative flex flex-1 flex-col items-center justify-center bg-white px-6 py-12 sm:px-10">
        <AuroraBackground
          aria-hidden
          showRadialGradient={false}
          className="absolute inset-0 h-full w-full lg:hidden"
        />
        <div className="relative z-10 flex w-full max-w-[26rem] flex-col">
          <AutoGPTLogo className="mx-auto mb-10 h-auto w-32 sm:w-40 lg:hidden" />
          {children}
        </div>
      </section>
    </div>
  );
}
