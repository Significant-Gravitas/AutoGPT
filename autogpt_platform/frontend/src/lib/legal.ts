export const TERMS_OF_USE_URL = "https://agpt.co/legal/platform-terms-of-use";
export const PRIVACY_POLICY_URL =
  "https://agpt.co/legal/platform-privacy-policy";
// Bump when the terms or the privacy policy change. Stored on the user as
// termsVersion so we can tell which text a given account agreed to. Format
// YYYY-MM (or YYYY-MM-DD for a second change in a month). The backend records
// only versions in RECOGNIZED_TERMS_VERSIONS (backend/api/model.py): add the
// new one there, and deploy it, before bumping this.
export const TERMS_VERSION = "2026-10";
