import type { MicrosoftCapabilityStatus } from "@/app/api/__generated__/models/microsoftCapabilityStatus";

export const MICROSOFT_PROVIDER = "microsoft";

// Mirror the backend's service catalog (microsoft_services.py). A first
// sign-in asks for the basics; Copilot's permissions only make sense on a work
// or school account, so they are asked for when someone connects for Copilot.
export const MICROSOFT_BASE_SCOPES = [
  "User.Read",
  "Mail.ReadWrite",
  "Mail.Send",
  "Calendars.ReadWrite",
  "Files.ReadWrite",
];
export const MICROSOFT_COPILOT_SCOPES = [
  "Sites.Read.All",
  "Mail.Read",
  "People.Read.All",
  "OnlineMeetingTranscript.Read.All",
  "Chat.Read",
  "ChannelMessage.Read.All",
  "ExternalItem.Read.All",
];

export const MICROSOFT_STATUS_LABELS: Record<
  MicrosoftCapabilityStatus,
  string
> = {
  available: "Ready",
  permission_needed: "Needs your permission",
  admin_required: "Needs your Microsoft admin's approval",
  setup_required: "Your organization still needs to set this up",
  work_account_only: "Work or school accounts only",
  unavailable: "Not available for this account",
};

export type MicrosoftSetupReason =
  | "tenant_setup_required"
  | "admin_consent_required"
  | "billing_setup_required"
  | "app_setup_required"
  | "access_denied"
  | "sign_in_failed";

export const MICROSOFT_SETUP_MESSAGES: Record<MicrosoftSetupReason, string> = {
  tenant_setup_required: "Your organization needs to enable Work IQ.",
  admin_consent_required:
    "Your Microsoft administrator needs to approve access.",
  billing_setup_required:
    "Your organization needs to finish Work IQ billing setup.",
  app_setup_required:
    "Browser sign-in needs configuration by the application owner.",
  access_denied:
    "Microsoft access was declined. Your existing connections are still available.",
  sign_in_failed:
    "Microsoft sign-in did not finish. Try again or open setup help.",
};

export const WORK_IQ_SETUP_URL =
  "https://learn.microsoft.com/en-us/microsoft-365/copilot/extensibility/work-iq/enable-work-iq";
export const WORK_IQ_PERMISSIONS_URL =
  "https://learn.microsoft.com/en-us/microsoft-365/copilot/extensibility/work-iq/permissions";
export const WORK_IQ_APP_ID = "fdcc1f02-fc51-4226-8753-f668596af7f7";

export function getWorkIQAdminConsentURL(tenantID: string): string | null {
  if (
    !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(
      tenantID,
    ) ||
    tenantID.toLowerCase() === "9188040d-6c67-4c5b-b112-36a304b66dad"
  )
    return null;
  return `https://login.microsoftonline.com/${tenantID}/adminconsent?client_id=${WORK_IQ_APP_ID}`;
}

export function getMicrosoftSetupReason(
  error: unknown,
): MicrosoftSetupReason | null {
  const detail = readDetail(error);
  if (detail && typeof detail === "object" && "reason" in detail) {
    const reason = detail.reason;
    if (
      typeof reason === "string" &&
      Object.hasOwn(MICROSOFT_SETUP_MESSAGES, reason)
    ) {
      return reason as MicrosoftSetupReason;
    }
  }
  const text =
    typeof detail === "string"
      ? detail
      : error instanceof Error
        ? error.message
        : typeof error === "string"
          ? error
          : "";
  if (/AADSTS650052\b/.test(text)) return "tenant_setup_required";
  if (/AADSTS(65001|90094|90093)\b|consent_required/.test(text))
    return "admin_consent_required";
  if (/AADSTS(50011|700016|7000218|7000215|50194)\b/.test(text))
    return "app_setup_required";
  if (/AADSTS65004\b|\baccess_denied\b|\bauthorization_declined\b/.test(text))
    return "access_denied";
  return null;
}

function readDetail(error: unknown): unknown {
  if (!error || typeof error !== "object") return null;
  if ("response" in error) return readDetail(error.response);
  if ("detail" in error) return error.detail;
  return null;
}

export function microsoftCallbackMessage(
  error: string,
  description: string,
): string {
  const reason =
    getMicrosoftSetupReason(`${error} ${description.slice(0, 4096)}`) ??
    "sign_in_failed";
  const code = description.match(/AADSTS\d+/)?.[0];
  return `${code ? `${code}: ` : ""}${MICROSOFT_SETUP_MESSAGES[reason]}`;
}
