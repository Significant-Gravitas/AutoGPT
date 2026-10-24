import { ReactNode } from "react";
import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { CookieSettingsLink } from "@/components/molecules/CookieSettingsLink/CookieSettingsLink";
import { cn } from "@/lib/utils";

interface Props {
  marketing: ReactNode;
  children: ReactNode;
  className?: string;
}

export function AuthSplitLayout({ marketing, children, className }: Props) {
  return (
    <div className={cn("flex min-h-screen w-full flex-1", className)}>
      <aside
        // TODO(tokens): no exact role for bg-[#f9f9f9] (#F9F9F9)
        className="relative hidden flex-1 border-r border-border-base bg-[#f9f9f9] lg:flex"
      >
        {marketing}
      </aside>
      <section className="relative flex flex-1 flex-col items-center bg-surface px-6 pb-4 pt-12 sm:px-10">
        <div className="flex w-full max-w-[26rem] flex-1 flex-col justify-center">
          <AutoGPTLogo className="mx-auto mb-10 h-auto w-32 sm:w-40 lg:hidden" />
          {children}
        </div>
        <CookieSettingsLink className="mt-6" />
      </section>
    </div>
  );
}
