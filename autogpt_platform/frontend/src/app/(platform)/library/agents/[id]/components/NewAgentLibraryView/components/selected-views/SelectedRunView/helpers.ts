import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";

const GOOGLE_PROVIDER = "google";

/** Whether the agent signs in to Google, so its runs can output the user's
 *  Gmail, Drive, Docs, Sheets or Calendar data. */
export function usesGoogleAccount(agent: LibraryAgent): boolean {
  const properties = agent.credentials_input_schema?.properties;
  if (!isRecord(properties)) return false;
  return Object.values(properties).some(
    (field) =>
      isRecord(field) &&
      Array.isArray(field.credentials_provider) &&
      field.credentials_provider.includes(GOOGLE_PROVIDER),
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
