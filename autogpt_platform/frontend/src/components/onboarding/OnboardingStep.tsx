"use client";
import { ReactNode } from "react";
import OnboardingBackButton from "./OnboardingBackButton";

export function OnboardingStep({ children }: { children: ReactNode }) {
  return (
    <div className="relative flex min-h-screen w-full flex-col">
      <div className="flex flex-col items-center">{children}</div>
    </div>
  );
}

interface OnboardingHeaderProps {
  backHref: string;
  children?: ReactNode;
}

export function OnboardingHeader({
  backHref,
  children,
}: OnboardingHeaderProps) {
  return (
    <div className="sticky top-0 z-10 w-full">
      <div className="bg-gray-100 pb-5">
        <div className="flex w-full items-center justify-between px-5 py-5">
          <OnboardingBackButton href={backHref} />
          <div>Progress...</div>
        </div>
        {children}
      </div>
      <div className="h-4 w-full bg-gradient-to-b from-gray-100 via-gray-100/50 to-transparent" />
    </div>
  );
}

export function OnboardingFooter({ children }: { children: ReactNode }) {
  return (
    <div className="sticky bottom-0 z-10 w-full">
      <div className="h-4 w-full bg-gradient-to-t from-gray-100 via-gray-100/50 to-transparent" />
      <div className="flex justify-center bg-gray-100">
        <div className="px-5 py-5">{children}</div>
      </div>
    </div>
  );
}
