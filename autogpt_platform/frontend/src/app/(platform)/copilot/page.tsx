"use client";

import { FeatureFlagPage } from "@/services/feature-flags/FeatureFlagPage";
import { Flag } from "@/services/feature-flags/use-get-flag";
import { CopilotPage } from "./CopilotPage";

export default function Page() {
  return (
    <FeatureFlagPage flag={Flag.CHAT} whenDisabled="/library">
      <CopilotPage />
    </FeatureFlagPage>
  );
}
