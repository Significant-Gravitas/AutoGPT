import type { ComponentType } from "react";
import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import { missingServices, pickGoogleCredential } from "@/lib/google-access";
import { MICROSOFT_BASE_SCOPES } from "@/lib/microsoft-setup";
import { GoogleAccess } from "./components/GoogleAccess/GoogleAccess";
import { MicrosoftAccess } from "./components/MicrosoftAccess";

export interface AccessConnectProps {
  onDone: () => void;
  /** Moves on without connecting; offered only until a connection exists. */
  onSkip: () => void;
}

interface AccessProvider {
  title: string;
  description: string;
  Connect: ComponentType<AccessConnectProps>;
  /** Whether the saved connections already hold everything the step offers,
   *  as a sign-in that connects (connect-at-sign-in.ts) leaves them. */
  isConnected: (credentials: readonly CredentialsMetaResponse[]) => boolean;
}

/**
 * Sign-in providers whose data Otto can also use, keyed by the provider id
 * the sign-in account is stored under. Signing up with one of these adds the
 * access step to onboarding, unless the sign-in already connected everything
 * the step offers; another provider (GitHub) joins by adding an entry here.
 */
export const ACCESS_PROVIDERS: Record<string, AccessProvider> = {
  microsoft: {
    title: "Also connect Microsoft 365?",
    description:
      "Signing in with Microsoft only proves who you are. Connecting is what lets Otto work with your Microsoft 365. You can change this anytime in Settings.",
    Connect: MicrosoftAccess,
    // Mail, calendar and files; Teams and Copilot stay in Settings.
    isConnected: (credentials) =>
      credentials.some(
        (credential) =>
          credential.type === "oauth2" &&
          MICROSOFT_BASE_SCOPES.every((scope) =>
            credential.scopes?.includes(scope),
          ),
      ),
  },
  google: {
    title: "Also connect Gmail, Drive and Calendar?",
    description:
      "Signing in with Google only proves who you are. Connecting is what lets Otto work with your email, files and calendar. You can change this anytime in Settings.",
    Connect: GoogleAccess,
    isConnected: (credentials) => {
      const credential = pickGoogleCredential(credentials);
      return !!credential && missingServices(credential).length === 0;
    },
  },
};
