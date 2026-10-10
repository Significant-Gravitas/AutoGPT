import type { SkillVersionFileSummary } from "@/app/api/__generated__/models/skillVersionFileSummary";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  file?: SkillVersionFileSummary;
  label: string;
}

export function FilePreview({ file, label }: Props) {
  if (!file) return null;
  return (
    <div className="min-w-0">
      <Text variant="small" tone="secondary">
        {label} · {file.size_bytes.toLocaleString()} bytes
        {file.is_executable ? " · Executable" : ""}
      </Text>
      {file.content != null ? (
        <pre className="mt-1 max-h-80 overflow-auto whitespace-pre-wrap break-words rounded-md bg-zinc-900 p-3 text-xs text-zinc-100">
          {file.content || "(empty file)"}
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
