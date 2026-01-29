"use client";
import { FeatureFlagPage } from "@/services/feature-flags/FeatureFlagPage";
import { Flag } from "@/services/feature-flags/use-get-flag";
import { type ReactNode } from "react";
import { CopilotShell } from "./components/CopilotShell/CopilotShell";

export default function CopilotLayout({ children }: { children: ReactNode }) {
  return (
    <FeatureFlagPage flag={Flag.CHAT} whenDisabled="/library">
      <CopilotShell>{children}</CopilotShell>
    </FeatureFlagPage>
  );
}
