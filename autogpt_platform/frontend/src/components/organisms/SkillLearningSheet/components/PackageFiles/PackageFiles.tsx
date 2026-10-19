import type { SkillVersionFileSummary } from "@/app/api/__generated__/models/skillVersionFileSummary";
import { Text } from "@/components/atoms/Text/Text";
import { FilePreview } from "./FilePreview";
import { packageFileChanges } from "./helpers";

interface Props {
  before?: SkillVersionFileSummary[] | null;
  after?: SkillVersionFileSummary[] | null;
}

export function PackageFiles({ before, after }: Props) {
  const changes = packageFileChanges(before, after);
  if (changes.length === 0) return null;
  return (
    <section
      aria-label="Package file changes"
      className="flex min-w-0 flex-col gap-2"
    >
      <Text variant="small-medium">Package file changes</Text>
      {changes.map((change) => (
        <details
          key={change.path}
          className="min-w-0 rounded-lg border border-zinc-200 p-3"
        >
          <summary className="cursor-pointer break-all text-sm">
            <span>{change.path}</span>
            {" · "}
            {!change.after ? "Removed" : change.before ? "Updated" : "Added"}
          </summary>
          <div className="mt-3 flex min-w-0 flex-col gap-3">
            <FilePreview file={change.before} label="Previous file" />
            <FilePreview file={change.after} label="New file" />
          </div>
        </details>
      ))}
    </section>
  );
}
