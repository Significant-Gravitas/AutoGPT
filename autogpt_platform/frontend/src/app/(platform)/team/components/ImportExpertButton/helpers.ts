import type { ExpertImportResult } from "@/app/api/__generated__/models/expertImportResult";
import { getFileSizeError } from "@/lib/direct-upload";

/** Matches `MAX_ZIP_BYTES` on the import route: a bigger file is rejected here
 *  rather than after a 20 MiB upload the server was always going to refuse. */
const MAX_PACKAGE_MB = 20;

export function getPackageFileError(file: File): string | null {
  if (!file.name.toLowerCase().endsWith(".zip")) {
    return "Pick a .expert.zip file — the one a download gives you.";
  }
  return getFileSizeError(file, MAX_PACKAGE_MB);
}

/** An import is partial by design, so the toast says what did not make it
 *  rather than pretending the whole file arrived. */
export function getImportFailureLine(
  result: ExpertImportResult,
): string | null {
  const failed = [
    ...(result.failed_workflows ?? []),
    ...(result.failed_skills ?? []),
  ];
  if (failed.length === 0) return null;

  const workflows = result.failed_workflows?.length ?? 0;
  const skills = result.failed_skills?.length ?? 0;
  const parts = [
    workflows > 0 ? `${workflows} workflow${workflows === 1 ? "" : "s"}` : null,
    skills > 0 ? `${skills} skill${skills === 1 ? "" : "s"}` : null,
  ].filter(Boolean);

  return `${parts.join(" and ")} couldn't be imported: ${failed.join(", ")}`;
}
