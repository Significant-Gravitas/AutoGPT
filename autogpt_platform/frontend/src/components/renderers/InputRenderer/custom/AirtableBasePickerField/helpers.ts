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

export type BaseAccess = "read" | "edit" | "create";

const ACCESS_RANK: Record<BaseAccess, number> = { read: 0, edit: 1, create: 2 };

// The least a base must allow for the step: Airtable lets only a base's
// creators change its tables, editors write records, and anyone read.
export function pickerAccess(schema: unknown): BaseAccess {
  if (!isRecord(schema) || !isRecord(schema.resource_picker)) return "create";
  const access = schema.resource_picker.access;
  return access === "read" || access === "edit" ? access : "create";
}

export function allows(access: BaseAccess, needed: BaseAccess): boolean {
  return ACCESS_RANK[access] >= ACCESS_RANK[needed];
}

// Why a base is listed but can't be picked for this step.
export function lockedReason(access: BaseAccess, needed: BaseAccess): string {
  if (access === "read") return "View only";
  return needed === "create" ? "Needs creator access" : "View only";
}

// What the step will do to the base, and only that: a table change is shown
// on a card first, a record change (adding, updating or deleting) is not, and
// a read changes nothing.
export function pickerHint(needed: BaseAccess): string {
  const base =
    "A base is Airtable's word for a set of tables, like a spreadsheet file with several sheets.";
  if (needed === "read") {
    return `${base} Otto only reads from the one you pick; nothing in it changes.`;
  }
  if (needed === "edit") {
    return `${base} Otto changes records in the one you pick.`;
  }
  return `${base} Otto adds what it needs to the one you pick, and you'll see what it adds before anything changes.`;
}

// What to do when no base allows the step.
export function noUsableBaseText(needed: BaseAccess): string {
  return needed === "create"
    ? "This step needs creator access to a base, which Airtable gives to the people who set it up. Ask the owner to make you a creator, or start a new one."
    : "You can only view these bases, so Otto can't change records in them. Ask the owner to make you an editor, or start a new one.";
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
