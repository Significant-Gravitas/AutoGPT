import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";

export const GOOGLE_PROVIDER = "google";

export interface GoogleService {
  name: string;
  /** What the permissions are for, shown before anyone grants them. */
  purpose: string;
  scopes: readonly string[];
}

/**
 * What connecting Google asks for, at sign-in (lib/auth/connect-at-sign-in.ts)
 * or in onboarding's access step, by the service each permission is for.
 * These are the scopes the Google blocks already ask for,
 * so one connection covers all of them and Google's consent screen lists
 * nothing the app isn't approved for. Gmail's metadata scope is never asked
 * for: a grant that includes it gets message headers only and no search
 * (`blocks/google/gmail.py`).
 */
export const GOOGLE_SERVICES: readonly GoogleService[] = [
  {
    name: "Gmail",
    purpose: "Lets Otto read, send, label and trash your email",
    scopes: [
      "https://www.googleapis.com/auth/gmail.readonly",
      "https://www.googleapis.com/auth/gmail.send",
      "https://www.googleapis.com/auth/gmail.modify",
      "https://www.googleapis.com/auth/gmail.labels",
    ],
  },
  {
    name: "Google Drive",
    purpose:
      "Lets Otto open and edit the Docs, Sheets and files you pick, and the ones it creates",
    scopes: ["https://www.googleapis.com/auth/drive.file"],
  },
  {
    name: "Google Calendar",
    purpose: "Lets Otto see, add and change your calendar events",
    scopes: [
      "https://www.googleapis.com/auth/calendar",
      "https://www.googleapis.com/auth/calendar.readonly",
    ],
  },
];

export const GOOGLE_ACCESS_SCOPES = GOOGLE_SERVICES.flatMap(
  (service) => service.scopes,
);

/** Whether the account granted every permission the service needs. */
export function isAllowed(
  service: GoogleService,
  granted: readonly string[] | null | undefined,
) {
  return service.scopes.every((scope) => granted?.includes(scope));
}

/** The services an account doesn't allow yet. */
export function missingServices(credential: CredentialsMetaResponse) {
  return GOOGLE_SERVICES.filter(
    (service) => !isAllowed(service, credential.scopes),
  );
}

/**
 * The connected Google account that already allows the most services. Google
 * lets people untick permissions on its consent screen, so an account can be
 * connected without all of them. Only a Google sign-in can be asked for more
 * permissions, so any other kind of credential saved under the provider is
 * never the account.
 */
export function pickGoogleCredential(
  credentials: readonly CredentialsMetaResponse[],
): CredentialsMetaResponse | undefined {
  let best: CredentialsMetaResponse | undefined;
  let bestCount = -1;
  for (const credential of credentials) {
    if (credential.type !== "oauth2") continue;
    const count = GOOGLE_SERVICES.filter((service) =>
      isAllowed(service, credential.scopes),
    ).length;
    if (count > bestCount) {
      best = credential;
      bestCount = count;
    }
  }
  return best;
}
