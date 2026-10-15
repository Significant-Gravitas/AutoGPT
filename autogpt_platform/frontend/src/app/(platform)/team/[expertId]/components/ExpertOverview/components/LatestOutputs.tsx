"use client";

import {
  File02Icon,
  Image01Icon,
  Table01Icon,
} from "@hugeicons/core-free-icons";
import { FileThumbnail } from "@/app/(platform)/artifacts/components/ArtifactsList/ArtifactsTable/FileThumbnail";
import {
  formatFileSize,
  formatRelativeDate,
  getFileDownloadUrl,
} from "@/app/(platform)/artifacts/components/ArtifactsList/helpers";
import { PanelEmptyState } from "@/app/(platform)/copilot/components/ContextPanel/components/PanelEmptyState/PanelEmptyState";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { OverviewCard } from "./OverviewCard";

const VISIBLE_OUTPUTS = 5;

interface Props {
  expertName: string;
  outputs: WorkspaceFileItem[];
  isLoading: boolean;
}

export function LatestOutputs({ expertName, outputs, isLoading }: Props) {
  return (
    <OverviewCard title="Latest outputs">
      {isLoading ? (
        <div className="flex flex-col gap-2">
          <Skeleton className="h-10 w-full rounded-lg" />
          <Skeleton className="h-10 w-full rounded-lg" />
        </div>
      ) : outputs.length === 0 ? (
        <PanelEmptyState
          tiles={[
            <Icon
              key="image"
              icon={Image01Icon}
              size={22}
              className="text-sky-500"
            />,
            <Icon
              key="table"
              icon={Table01Icon}
              size={22}
              className="text-emerald-500"
            />,
            <Icon
              key="doc"
              icon={File02Icon}
              size={22}
              className="text-violet-500"
            />,
          ]}
          title="No outputs yet"
          description={`Documents and files ${expertName} creates show up here.`}
        />
      ) : (
        <ul className="-mx-2 flex flex-col">
          {outputs.slice(0, VISIBLE_OUTPUTS).map((file) => (
            <li key={file.id}>
              <a
                href={getFileDownloadUrl(file.id)}
                target="_blank"
                rel="noreferrer"
                title={file.name}
                className="flex items-center gap-3 rounded-xl px-2 py-2 transition-colors hover:bg-zinc-50"
              >
                <FileThumbnail file={file} />
                <span className="flex min-w-0 flex-1 flex-col">
                  <Text
                    variant="body-medium"
                    as="span"
                    className="truncate !text-zinc-900"
                  >
                    {file.name}
                  </Text>
                  <Text variant="small" as="span" className="!text-zinc-500">
                    {formatRelativeDate(file.created_at)} ·{" "}
                    {formatFileSize(file.size_bytes)}
                  </Text>
                </span>
              </a>
            </li>
          ))}
        </ul>
      )}
    </OverviewCard>
  );
}
