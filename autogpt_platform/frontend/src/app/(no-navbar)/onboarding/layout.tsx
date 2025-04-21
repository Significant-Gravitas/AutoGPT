import { ReactNode, Suspense } from "react";

export default function OnboardingLayout({
  children,
}: {
  children: ReactNode;
}) {
  return (
    <div className="flex min-h-screen w-full items-center justify-center bg-gray-100">
      <div className="mx-auto flex w-full flex-col items-center">
        <main className="w-full flex-grow">
          <Suspense>{children}</Suspense>
        </main>
      </div>
    </div>
  );
}
