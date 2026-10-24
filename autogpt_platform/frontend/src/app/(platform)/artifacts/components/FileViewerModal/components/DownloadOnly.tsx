import { Icon } from "@/components/atoms/Icon/Icon";
import { Download04Icon } from "@hugeicons/core-free-icons";

interface Props {
  name: string;
  downloadUrl: string;
  message?: string;
}

export function DownloadOnly({
  name,
  downloadUrl,
  message = "This file type can't be previewed.",
}: Props) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
      <p className="text-sm text-foreground-tertiary">{message}</p>
      <a
        href={downloadUrl}
        download={name}
        className="inline-flex items-center gap-1.5 rounded-md border border-border-base bg-surface px-3 py-1.5 text-sm font-medium text-foreground-body transition-colors hover:bg-surface-muted"
      >
        <Icon icon={Download04Icon} size={16} />
        Download
      </a>
    </div>
  );
}
