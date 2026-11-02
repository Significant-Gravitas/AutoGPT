import type { SkillPackageFile } from "@/app/api/__generated__/models/skillPackageFile";
import { Icon } from "@/components/atoms/Icon/Icon";
import { File01Icon } from "@hugeicons/core-free-icons";
import { formatFileSize } from "./helpers";

interface Props {
  files: SkillPackageFile[];
  onOpenFile: (path: string) => void;
}

export function SkillPackageFileList({ files, onOpenFile }: Props) {
  return (
    <ul
      className="divide-y divide-separator overflow-hidden rounded-xl border border-border-base"
      data-testid="skill-package-files"
    >
      {files.map((file) => (
        <li key={file.path}>
          <button
            type="button"
            onClick={() => onOpenFile(file.path)}
            className="flex w-full items-center gap-3 px-3 py-2 text-left transition-colors hover:bg-surface-muted focus-visible:ring-2 focus-visible:ring-border-strong focus-visible:-outline-offset-2 focus-visible:outline-none"
          >
            <Icon
              icon={File01Icon}
              size={14}
              className="shrink-0 text-foreground-subtle"
              aria-hidden
            />
            <span className="min-w-0 flex-1 truncate font-mono text-body-sm leading-inherit text-foreground-body">
              {file.path}
            </span>
            {file.is_executable ? (
              <span className="shrink-0 rounded-md bg-surface-hover px-1.5 py-0.5 text-caption-sm leading-inherit font-medium text-foreground-tertiary">
                executable
              </span>
            ) : null}
            <span className="shrink-0 text-xs text-foreground-subtle tabular-nums">
              {formatFileSize(file.size_bytes)}
            </span>
          </button>
        </li>
      ))}
    </ul>
  );
}
