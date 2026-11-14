"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { formatTimeAgo } from "@/lib/utils/time";
import { File02Icon } from "@hugeicons/core-free-icons";
import { formatFileSize } from "../ContextPanel/components/FilesTab/helpers";
import { useWorkspaceFileCards } from "../WorkspaceFileCards/useWorkspaceFileCards";

interface Props {
  sessionId: string;
}

/** The files of this chat as a plain list: a click opens one in a tab of
 *  its own. */
export function FilesTab({ sessionId }: Props) {
  const { files, isLoading, isError, handleOpen } =
    useWorkspaceFileCards(sessionId);

  if (isLoading) {
    return (
      <Text variant="inbox" tone="faint" className="m-1.5">
        Loading the files…
      </Text>
    );
  }
  if (isError) {
    return (
      <Text variant="inbox" tone="faint" className="m-1.5">
        The files could not be loaded.
      </Text>
    );
  }
  if (files.length === 0) {
    return (
      <Text variant="inbox" tone="faint" className="m-1.5">
        No files in this chat yet.
      </Text>
    );
  }

  return (
    <ul className="flex flex-col">
      {files.map((file) => (
        <li key={file.item.id}>
          <button
            type="button"
            onClick={() => handleOpen(file)}
            className="flex h-9 w-full items-center gap-2.5 rounded-lg px-1.5 text-left transition-colors duration-100 ease-out outline-none hover:bg-surface-wash focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:ring-inset"
          >
            <Icon icon={File02Icon} size={14} tone="quiet" />
            <Text
              as="span"
              variant="inbox"
              tone="ink"
              className="min-w-0 flex-1 truncate"
            >
              {file.item.name}
            </Text>
            <Text
              as="span"
              variant="inbox"
              tone="faint"
              tabular
              className="shrink-0"
            >
              {formatFileSize(file.item.size_bytes)}
            </Text>
            <Text as="span" variant="inbox" tone="faint" className="shrink-0">
              {formatTimeAgo(file.item.created_at)}
            </Text>
          </button>
        </li>
      ))}
    </ul>
  );
}
