import type { ComponentType } from "react";
import { MicrosoftAccess } from "./components/MicrosoftAccess";

export interface AccessConnectProps {
  onDone: () => void;
}

interface AccessProvider {
  title: string;
  description: string;
  Connect: ComponentType<AccessConnectProps>;
}

/**
 * Sign-in providers whose data Otto can also use, keyed by the provider id
 * the sign-in account is stored under. Signing up with one of these adds the
 * access step to onboarding; another provider (Google, GitHub) joins by
 * adding an entry here.
 */
export const ACCESS_PROVIDERS: Record<string, AccessProvider> = {
  microsoft: {
    title: "Also connect Microsoft 365?",
    description:
      "Otto can then help with your mail, calendar and files. You can change this anytime in Settings.",
    Connect: MicrosoftAccess,
  },
};
