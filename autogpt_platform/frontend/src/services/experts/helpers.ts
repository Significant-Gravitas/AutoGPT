const NON_SLUG_RUN = /[^a-z0-9]+/g;
const EDGE_DASHES = /^-+|-+$/g;
const MAX_SLUG_LENGTH = 60;

/** The filename a download falls back to. Mirrors the backend's `expert_slug`,
 *  so the guess and the `Content-Disposition` agree on one name. */
export function expertPackageFilename(name: string): string {
  const slug = name
    .trim()
    .toLowerCase()
    .replace(NON_SLUG_RUN, "-")
    .replace(EDGE_DASHES, "")
    .slice(0, MAX_SLUG_LENGTH)
    .replace(EDGE_DASHES, "");

  return `${slug || "expert"}.expert.zip`;
}
