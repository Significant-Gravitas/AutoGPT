"use client";

import { FeatureFlagRedirect } from "@/services/feature-flags/FeatureFlagRedirect";
import { Flag } from "@/services/feature-flags/use-get-flag";

export default function Page() {
  return (
    <FeatureFlagRedirect
      flag={Flag.CHAT}
      whenEnabled="/copilot"
      whenDisabled="/library"
    />
  );
}
