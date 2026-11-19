import { APIError } from "better-auth/api";
import {
  PROFILE_IMAGE_URL_MAX_LENGTH,
  PROFILE_NAME_MAX_LENGTH,
} from "./profile-field-lengths";

export const INVALID_PROFILE_FIELD_CODE = "INVALID_PROFILE_FIELD";

const MAX_LENGTHS = {
  name: PROFILE_NAME_MAX_LENGTH,
  preferredName: PROFILE_NAME_MAX_LENGTH,
  image: PROFILE_IMAGE_URL_MAX_LENGTH,
};

// Better Auth takes any JSON value for these, and the database would store an
// array, object or number as its text, so only strings within the limit pass.
export function assertProfileFieldsFit(fields: unknown) {
  if (typeof fields !== "object" || fields === null) return;
  for (const [field, maxLength] of Object.entries(MAX_LENGTHS)) {
    const value = (fields as Record<string, unknown>)[field];
    if (value === undefined || value === null) continue;
    if (typeof value === "string" && value.length <= maxLength) continue;
    throw new APIError("BAD_REQUEST", {
      code: INVALID_PROFILE_FIELD_CODE,
      message: `${field} must be text of at most ${maxLength} characters`,
    });
  }
}
