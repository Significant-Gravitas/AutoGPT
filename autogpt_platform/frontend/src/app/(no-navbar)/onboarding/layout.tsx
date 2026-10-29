import { ReactNode } from "react";

export default function OnboardingLayout({
  children,
}: {
  children: ReactNode;
}) {
  return (
    <div
      // TODO(tokens): no exact role for bg-surface-hover (#F3F4F6)
      className="relative flex min-h-screen w-full flex-col bg-surface-hover"
    >
      <main className="flex w-full flex-1 flex-col items-center justify-center">
        {children}
      </main>
    </div>
  );
}
