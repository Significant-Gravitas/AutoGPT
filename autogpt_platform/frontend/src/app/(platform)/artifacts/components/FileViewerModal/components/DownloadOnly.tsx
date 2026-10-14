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
      <p className="text-sm text-zinc-500">{message}</p>
      <a
        href={downloadUrl}
        download={name}
        className="inline-flex items-center gap-1.5 rounded-md border border-zinc-200 bg-white px-3 py-1.5 text-sm font-medium text-zinc-700 transition-colors hover:bg-zinc-50"
      >
        <Icon icon={Download04Icon} size={16} />
        Download
      </a>
    </div>
  );
}
