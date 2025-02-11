'use client';
import { ReactNode } from "react";
import OnboardingBackButton from "./OnboardingBackButton";

interface OnboardingStepProps {
  backHref: string;
  children?: ReactNode;
}

export default function OnboardingStep({ backHref, children }: OnboardingStepProps) {



  return (
    <div className="w-full min-h-screen flex flex-col">
      <div className="w-full px-5 py-5 flex justify-between items-center">
        <OnboardingBackButton href={backHref} />
        <div>Progress...</div>
      </div>
      <div className="flex flex-col items-center">
        {children}
      </div>
    </div>
  );
}
