"use client";

import {
  File02Icon,
  Image01Icon,
  Table01Icon,
} from "@hugeicons/core-free-icons";
import { PanelEmptyState } from "@/app/(platform)/copilot/components/ContextPanel/components/PanelEmptyState/PanelEmptyState";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { ExpertFileRow } from "../../ExpertFilesSection/ExpertFileRow";
import { CardError } from "./CardError";
import { OverviewCard } from "./OverviewCard";

const VISIBLE_OUTPUTS = 5;

interface Props {
  expertName: string;
  outputs: WorkspaceFileItem[];
  isLoading: boolean;
  isError: boolean;
  onRetry: () => unknown;
}

export function LatestOutputs({
  expertName,
  outputs,
  isLoading,
  isError,
  onRetry,
}: Props) {
  return (
    <OverviewCard title="Latest outputs">
      {isLoading ? (
        <div className="flex flex-col gap-2">
          <Skeleton className="h-10 w-full rounded-lg" />
          <Skeleton className="h-10 w-full rounded-lg" />
        </div>
      ) : isError ? (
        <CardError message="Couldn't load outputs." onRetry={onRetry} />
      ) : outputs.length === 0 ? (
        <PanelEmptyState
          textSize="body-md"
          tiles={[
            // oxlint-disable-next-line design-system/no-raw-colors -- sky has no role; info blue would change the hue (deferred)
            <span key="image" className="inline-flex text-sky-500">
              <Icon icon={Image01Icon} size={22} />
            </span>,
            <span key="table" className="inline-flex text-success">
              <Icon icon={Table01Icon} size={22} />
            </span>,
            <span key="doc" className="inline-flex text-brand">
              <Icon icon={File02Icon} size={22} />
            </span>,
          ]}
          title="No outputs yet"
          description={`Documents and files ${expertName} creates show up here.`}
        />
      ) : (
        <ul className="-mx-2 flex flex-col">
          {outputs.slice(0, VISIBLE_OUTPUTS).map((file) => (
            <li key={file.id}>
              <ExpertFileRow file={file} />
            </li>
          ))}
        </ul>
      )}
    </OverviewCard>
  );
}
