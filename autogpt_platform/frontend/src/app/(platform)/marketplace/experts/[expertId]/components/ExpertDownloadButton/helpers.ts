import { ApiError } from "@/lib/autogpt-server-api/helpers";

const PACKAGE_SUFFIX = ".expert.zip";
const SLUG_MAX_CHARS = 60;

const ERROR_DESCRIPTIONS: Record<number, string> = {
  401: "Your session has ended. Sign in again to download it.",
  404: "This expert is no longer on the marketplace.",
  413: "This expert is too large to download as a file.",
};

const FALLBACK_ERROR_DESCRIPTION =
  "Something went wrong. Try again in a moment.";

export function getPackageFilename(
  contentDisposition: string | null,
  expertName: string,
) {
  const fromHeader = contentDisposition?.match(/filename="([^"]+)"/)?.[1];
  return fromHeader ?? `${slugify(expertName) || "expert"}${PACKAGE_SUFFIX}`;
}

export function getDownloadErrorDescription(error: unknown) {
  const status = error instanceof ApiError ? error.status : null;
  return (
    (status !== null ? ERROR_DESCRIPTIONS[status] : undefined) ??
    FALLBACK_ERROR_DESCRIPTION
  );
}

function slugify(text: string) {
  return text
    .normalize("NFKD")
    .replace(/[^\x00-\x7f]/g, "")
    .toLowerCase()
    .replace(/'/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .slice(0, SLUG_MAX_CHARS)
    .replace(/^-+|-+$/g, "");
}
