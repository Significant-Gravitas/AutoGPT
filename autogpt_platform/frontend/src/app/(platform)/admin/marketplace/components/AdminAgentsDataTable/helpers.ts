import { StoreSubmission } from "@/lib/autogpt-server-api/types";

export function getLatestVersionByNumber(
  versions: StoreSubmission[],
): StoreSubmission | null {
  if (!versions || versions.length === 0) return null;
  return versions.reduce(
    (latest, current) =>
      (current.version ?? 0) > (latest.version ?? 1) ? current : latest,
    versions[0],
  );
}
