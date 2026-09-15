"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Copy01Icon, Download01Icon } from "@hugeicons/core-free-icons";

interface Props {
  url: string;
  onCopy: () => void;
  onDownload: () => void;
}

export function AvatarResultPanel({ url, onCopy, onDownload }: Props) {
  return (
    <section className="flex flex-col gap-3 rounded-2xl border border-zinc-200 bg-white p-4">
      <Text variant="body-medium" as="h2">
        Avatar URL
      </Text>
      <code
        data-testid="avatar-url"
        className="block overflow-x-auto whitespace-nowrap rounded-lg bg-zinc-100 px-3 py-2 font-mono text-sm text-zinc-800"
      >
        {url}
      </code>
      <div className="flex flex-wrap gap-2">
        <Button
          variant="secondary"
          size="small"
          className="rounded-full"
          leftIcon={<Icon icon={Copy01Icon} size={16} />}
          onClick={onCopy}
        >
          Copy URL
        </Button>
        <Button
          variant="secondary"
          size="small"
          className="rounded-full"
          leftIcon={<Icon icon={Download01Icon} size={16} />}
          onClick={onDownload}
        >
          Download SVG
        </Button>
      </div>
      <Text variant="small" tone="muted">
        The same URL renders server-side, so it works anywhere an image does.
      </Text>
    </section>
  );
}
