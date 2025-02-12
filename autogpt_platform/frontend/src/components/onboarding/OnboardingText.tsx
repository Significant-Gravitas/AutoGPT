import { ReactNode } from "react";

export function OnboardingText({
  className,
  isHeader,
  children,
}: {
  className?: string;
  isHeader?: boolean;
  children: ReactNode;
}) {
  return (
    <div
      className={`${className} font-poppins text-center ${
        isHeader
          ? "text-xl font-medium leading-7 text-zinc-900"
          : "text-sm font-normal leading-6 text-zinc-500"
      }`}
    >
      {children}
    </div>
  );
}
