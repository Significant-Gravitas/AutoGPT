import type { SkillVersionFileSummary } from "@/app/api/__generated__/models/skillVersionFileSummary";

export function packageFileChanges(
  before: SkillVersionFileSummary[] | null | undefined,
  after: SkillVersionFileSummary[] | null | undefined,
) {
  if (!after) return [];
  const previous = new Map(before?.map((file) => [file.relative_path, file]));
  const next = new Map(after.map((file) => [file.relative_path, file]));
  return Array.from(new Set([...previous.keys(), ...next.keys()]))
    .sort()
    .map((path) => ({
      path,
      before: previous.get(path),
      after: next.get(path),
    }))
    .filter(
      (change) =>
        change.before?.sha256 !== change.after?.sha256 ||
        change.before?.is_executable !== change.after?.is_executable,
    );
}
