import { APIError } from "better-auth/api";
import {
  PROFILE_EMAIL_MAX_LENGTH,
  PROFILE_IMAGE_URL_MAX_LENGTH,
  PROFILE_NAME_MAX_LENGTH,
} from "./profile-field-lengths";

export const INVALID_PROFILE_FIELD_CODE = "INVALID_PROFILE_FIELD";

const MAX_LENGTHS: Record<string, number> = {
  name: PROFILE_NAME_MAX_LENGTH,
  preferredName: PROFILE_NAME_MAX_LENGTH,
  image: PROFILE_IMAGE_URL_MAX_LENGTH,
};

// The stored address too. A request's `email` is also a sign-in address, which
// the attempt limits count whatever its length, so only the row checks it.
export const USER_ROW_MAX_LENGTHS: Record<string, number> = {
  ...MAX_LENGTHS,
  email: PROFILE_EMAIL_MAX_LENGTH,
};

// Only text within each field's limit passes; null clears a field.
export function assertProfileFieldsFit(
  fields: unknown,
  maxLengths = MAX_LENGTHS,
) {
  if (typeof fields !== "object" || fields === null) return;
  for (const [field, maxLength] of Object.entries(maxLengths)) {
    const value = (fields as Record<string, unknown>)[field];
    if (value === undefined || value === null) continue;
    if (typeof value === "string" && value.length <= maxLength) continue;
    throw new APIError("BAD_REQUEST", {
      code: INVALID_PROFILE_FIELD_CODE,
      message: `${field} must be text of at most ${maxLength} characters`,
    });
  }
}
