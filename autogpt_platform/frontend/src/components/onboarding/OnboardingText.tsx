import { ReactNode } from "react";

export function OnboardingText({ className, isHeader, children }: { className?: string, isHeader?: boolean, children: ReactNode }) {
  return (
    <div className={`${className} font-poppins text-center
                    ${isHeader ?
                      'text-xl font-medium text-zinc-900 leading-7'
                      : 'text-sm font-normal text-zinc-500 leading-6'}`}>
      {children}
    </div>
  );
}