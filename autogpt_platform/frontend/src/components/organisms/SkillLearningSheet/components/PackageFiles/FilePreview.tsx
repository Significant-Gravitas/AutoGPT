import type { SkillVersionFileSummary } from "@/app/api/__generated__/models/skillVersionFileSummary";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  file?: SkillVersionFileSummary;
  label: string;
}

export function FilePreview({ file, label }: Props) {
  if (!file) return null;
  return (
    <div data-slot="skill-learning-sheet-file-preview" className="min-w-0">
      <Text variant="small" tone="secondary">
        {label} · {file.size_bytes.toLocaleString()} bytes
        {file.is_executable ? " · Executable" : ""}
      </Text>
      {file.content != null ? (
        <pre
          data-slot="skill-learning-sheet-file-preview-content"
          className="mt-1 max-h-80 overflow-auto whitespace-pre-wrap break-words rounded-md bg-action-hover p-3 text-xs text-action-foreground"
        >
          {file.content ||
            (file.content_truncated ? "(preview unavailable)" : "(empty file)")}
        </pre>
      ) : (
        <Text variant="small" tone="muted">
          Binary file
        </Text>
      )}
      {file.content_truncated ? (
        <Text variant="small" tone="muted">
          Preview truncated; the complete file is retained in this version.
        </Text>
      ) : null}
    </div>
  );
}
