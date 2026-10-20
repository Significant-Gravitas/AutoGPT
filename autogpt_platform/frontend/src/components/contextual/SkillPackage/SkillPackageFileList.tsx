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
      className="divide-y divide-zinc-100 overflow-hidden rounded-xl border border-zinc-200"
      data-testid="skill-package-files"
    >
      {files.map((file) => (
        <li key={file.path}>
          <button
            type="button"
            onClick={() => onOpenFile(file.path)}
            className="flex w-full items-center gap-3 px-3 py-2 text-left transition-colors hover:bg-zinc-50 focus-visible:outline-none focus-visible:-outline-offset-2 focus-visible:ring-2 focus-visible:ring-zinc-300"
          >
            <Icon
              icon={File01Icon}
              size={14}
              className="shrink-0 text-zinc-400"
              aria-hidden
            />
            <span className="min-w-0 flex-1 truncate font-mono text-[13px] text-zinc-700">
              {file.path}
            </span>
            {file.is_executable ? (
              <span className="shrink-0 rounded-md bg-zinc-100 px-1.5 py-0.5 text-[11px] font-medium text-zinc-500">
                executable
              </span>
            ) : null}
            <span className="shrink-0 text-xs tabular-nums text-zinc-400">
              {formatFileSize(file.size_bytes)}
            </span>
          </button>
        </li>
      ))}
    </ul>
  );
}
