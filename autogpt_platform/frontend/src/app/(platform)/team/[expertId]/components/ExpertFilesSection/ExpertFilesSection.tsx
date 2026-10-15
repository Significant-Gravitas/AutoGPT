"use client";

import {
  File02Icon,
  Folder01Icon,
  Image01Icon,
  SparklesIcon,
  Table01Icon,
  Upload04Icon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { PillFilterMenu } from "../../../components/PillFilterMenu";
import { SectionEmptyState } from "../../../components/SectionEmptyState";
import { ToolbarSearch } from "../../../components/ToolbarSearch";
import { ExpertFileRow } from "./ExpertFileRow";
import { useExpertFilesSection } from "./useExpertFilesSection";

const FILTERS = [
  { value: "all", label: "All files", icon: Folder01Icon },
  { value: "generated", label: "Created", icon: SparklesIcon },
  { value: "uploaded", label: "Uploaded", icon: Upload04Icon },
] as const;

interface Props {
  expertId: string;
  expertName: string;
}

/** Everything in the expert's chats: what it made and what you gave it. */
export function ExpertFilesSection({ expertId, expertName }: Props) {
  const files = useExpertFilesSection(expertId);

  return (
    <section>
      <div className="mb-2.5 flex flex-wrap items-center justify-between gap-3">
        <Text variant="large-medium" tone="primary">
          {expertName}&apos;s Files
        </Text>
        <div className="flex w-full flex-wrap items-center gap-2 sm:w-auto">
          <ToolbarSearch
            id="expert-files-search"
            label="Search files"
            value={files.query}
            onChange={files.setQuery}
          />
          <PillFilterMenu
            label="Filter files"
            value={files.filter}
            defaultValue="all"
            options={FILTERS}
            onChange={files.setFilter}
          />
        </div>
      </div>

      {files.isLoading ? (
        <div className="mt-4 flex flex-col gap-2">
          <Skeleton className="h-14 w-full rounded-xl" />
          <Skeleton className="h-14 w-full rounded-xl" />
          <Skeleton className="h-14 w-full rounded-xl" />
        </div>
      ) : files.isError ? (
        <ErrorCard
          context={`${expertName}'s files`}
          hint="We could not load the files."
          onRetry={() => files.refetch()}
        />
      ) : files.files.length === 0 ? (
        <SectionEmptyState
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
          title="No files yet"
          description={`Files you share with ${expertName} and documents it creates show up here.`}
        />
      ) : files.visible.length === 0 ? (
        <Text variant="body" tone="muted" className="pt-4">
          No files match.
        </Text>
      ) : (
        <div className="mt-4 rounded-2xl bg-white p-2 smooth-shadow-ring-sm">
          <ul aria-label={`${expertName}'s files`} className="flex flex-col">
            {files.visible.map((file) => (
              <li key={file.id}>
                <ExpertFileRow file={file} />
              </li>
            ))}
          </ul>
          {files.hasMore ? (
            <Text variant="small" className="px-2 pb-1 pt-2 !text-zinc-500">
              Showing the latest 200 files.
            </Text>
          ) : null}
        </div>
      )}
    </section>
  );
}
