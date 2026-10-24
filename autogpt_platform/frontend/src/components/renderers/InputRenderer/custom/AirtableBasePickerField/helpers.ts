import { ApiError } from "@/lib/autogpt-server-api/helpers";

export const AIRTABLE_PICKER_FORMAT = "airtable-base-picker";
export const AIRTABLE_HOME_URL = "https://airtable.com/";

// The search box appears once the list is too long to scan.
export const SEARCH_THRESHOLD = 8;

// The copilot fills in the account the run will use; without one (the builder)
// there is nothing to list, so the field stays a plain text input.
export function pickerCredentialsID(schema: unknown): string | null {
  if (!isRecord(schema) || schema.format !== AIRTABLE_PICKER_FORMAT) {
    return null;
  }
  const picker = schema.resource_picker;
  if (!isRecord(picker)) return null;
  const id = picker.credentials_id;
  return typeof id === "string" && id ? id : null;
}

// Only a block that changes the base needs one the user can edit; a read
// works on a view-only base too.
export function pickerEditableOnly(schema: unknown): boolean {
  if (!isRecord(schema) || !isRecord(schema.resource_picker)) return true;
  return schema.resource_picker.editable_only !== false;
}

export function isAirtableBasePickerSchema(schema: unknown): boolean {
  return pickerCredentialsID(schema) !== null;
}

// The server words these for the user: reconnect, busy, or try again.
const WORDED_STATUSES = new Set([409, 502, 503]);

export function loadErrorText(error: unknown): string {
  if (error instanceof ApiError && WORDED_STATUSES.has(error.status)) {
    return error.message;
  }
  return "Couldn't load your Airtable bases. Try again.";
}

export function matchesSearch(name: string, search: string): boolean {
  return name.toLocaleLowerCase().includes(search.trim().toLocaleLowerCase());
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
