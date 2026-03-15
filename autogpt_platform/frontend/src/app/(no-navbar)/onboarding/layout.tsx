import { ReactNode } from "react";

export default function OnboardingLayout({
  children,
}: {
  children: ReactNode;
}) {
  return (
    <div className="flex min-h-screen w-full items-center justify-center bg-gray-100">
      <main className="mx-auto flex w-full flex-col items-center">
        {children}
      </main>
    </div>
  );
}
