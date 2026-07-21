import { ReactNode } from "react";

export default function OnboardingLayout({
  children,
}: {
  children: ReactNode;
}) {
  return (
    <div className="relative flex min-h-screen w-full flex-col bg-gray-100">
      <main className="flex w-full flex-1 flex-col items-center justify-center">
        {children}
      </main>
    </div>
  );
}
