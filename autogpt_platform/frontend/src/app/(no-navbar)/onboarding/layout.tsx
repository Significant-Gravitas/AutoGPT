import { FeatureFlagPage } from "@/services/feature-flags/FeatureFlagPage";
import { Flag } from "@/services/feature-flags/use-get-flag";
import { ReactNode } from "react";

export default function OnboardingLayout({
  children,
}: {
  children: ReactNode;
}) {
  return (
    <FeatureFlagPage flag={Flag.CHAT} whenDisabled="/library">
      <div className="flex min-h-screen w-full items-center justify-center bg-gray-100">
        <main className="mx-auto flex w-full flex-col items-center">
          {children}
        </main>
      </div>
    </FeatureFlagPage>
  );
}
