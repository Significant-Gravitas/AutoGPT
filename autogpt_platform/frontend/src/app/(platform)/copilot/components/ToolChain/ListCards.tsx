"use client";

import {
  BookOpenIcon,
  ClockIcon,
  FileIcon,
  FolderIcon,
  ImageIcon,
  LinkSquare01Icon,
  RepeatIcon,
} from "@hugeicons/core-free-icons";
import Link from "next/link";
import { Icon } from "@/components/atoms/Icon/Icon";
import { CARD, HALF, RESULT_GRID } from "./ResultCards";
import {
  formatBytes,
  formatWhen,
  inline,
  resultItemKey,
  str,
} from "./resultHelpers";

interface SchedulesProps {
  schedules: Record<string, unknown>[];
}

interface OutputProps {
  output: Record<string, unknown>;
}

interface FoldersProps {
  folders: Record<string, unknown>[];
}

interface FilesProps {
  files: Record<string, unknown>[];
}

interface ResultsProps {
  results: Record<string, unknown>[];
}

export function FeatureRequestList({ results }: ResultsProps) {
  return (
    <div className={RESULT_GRID}>
      {results.map((request, i) => (
        <div
          key={resultItemKey(request, i)}
          className={CARD + " flex items-start gap-2.5 p-2.5"}
        >
          <div className="min-w-0 flex-1">
            <p className="truncate text-body-sm font-medium leading-inherit text-foreground-emphasis">
              {str(request, "title") ?? inline(request)}
            </p>
            {str(request, "description") && (
              <p className="truncate text-xs text-foreground-tertiary">
                {str(request, "description")}
              </p>
            )}
          </div>
          {str(request, "identifier") && (
            <span className="shrink-0 rounded-full bg-surface-hover px-2 py-0.5 text-caption-sm leading-inherit text-foreground-tertiary">
              {str(request, "identifier")}
            </span>
          )}
        </div>
      ))}
    </div>
  );
}

export function ScheduleList({ schedules }: SchedulesProps) {
  return (
    <div className={RESULT_GRID}>
      {schedules.map((schedule, i) => {
        const next = str(schedule, "next_run_time");
        const recurring = !!str(schedule, "cron");
        return (
          <div
            key={resultItemKey(schedule, i)}
            className={CARD + " flex items-center gap-2.5 p-2.5"}
          >
            <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-surface-hover">
              <Icon
                icon={ClockIcon}
                size={15}
                className="text-foreground-secondary"
              />
            </div>
            <div className="min-w-0 flex-1">
              <p className="truncate text-body-sm font-medium leading-inherit text-foreground-emphasis">
                {str(schedule, "name", "message") ?? inline(schedule)}
              </p>
              {next && (
                <p className="flex items-center gap-1 text-xs text-foreground-tertiary">
                  {recurring && <Icon icon={RepeatIcon} size={11} />}
                  {formatWhen(next)}
                </p>
              )}
            </div>
            {str(schedule, "kind") && (
              <span className="shrink-0 rounded-full bg-surface-hover px-2 py-0.5 text-caption-sm leading-inherit text-foreground-tertiary">
                {str(schedule, "kind") === "copilot_turn" ? "chat" : "workflow"}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}

export function ScheduleCreatedCard({ output }: OutputProps) {
  const next = str(output, "next_run_time");
  return (
    <div className={`${CARD} ${HALF} flex items-center gap-2.5 p-2.5`}>
      <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-surface-hover">
        <Icon
          icon={ClockIcon}
          size={15}
          className="text-foreground-secondary"
        />
      </div>
      <div className="min-w-0 flex-1">
        <p className="truncate text-body-sm font-medium leading-inherit text-foreground-emphasis">
          Follow-up scheduled
        </p>
        {next && (
          <p className="flex items-center gap-1 text-xs text-foreground-tertiary">
            {output.is_recurring === true && (
              <Icon icon={RepeatIcon} size={11} />
            )}
            {formatWhen(next)}
          </p>
        )}
      </div>
    </div>
  );
}

export function FolderList({ folders }: FoldersProps) {
  return (
    <div className={RESULT_GRID}>
      {folders.map((folder, i) => {
        const count =
          typeof folder.agent_count === "number" ? folder.agent_count : null;
        return (
          <div
            key={resultItemKey(folder, i)}
            className={CARD + " flex items-center gap-2.5 p-2.5"}
          >
            <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-surface-hover">
              <Icon
                icon={FolderIcon}
                size={15}
                className="text-foreground-secondary"
              />
            </div>
            <p className="min-w-0 flex-1 truncate text-body-sm font-medium leading-inherit text-foreground-emphasis">
              {str(folder, "name") ?? inline(folder)}
            </p>
            {count !== null && (
              <span className="shrink-0 text-xs text-foreground-subtle">
                {count} workflow{count === 1 ? "" : "s"}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}

export function FileList({ files }: FilesProps) {
  return (
    <div className={`${CARD} ${HALF} divide-y divide-separator`}>
      {files.map((file, i) => {
        const mime = str(file, "mime_type") ?? "";
        const size =
          typeof file.size_bytes === "number" ? file.size_bytes : null;
        const fileIcon = mime.startsWith("image/") ? ImageIcon : FileIcon;
        return (
          <div
            key={resultItemKey(file, i)}
            className="flex items-center gap-2.5 px-2.5 py-2"
          >
            <Icon
              icon={fileIcon}
              size={14}
              className="shrink-0 text-foreground-subtle"
            />
            <p className="min-w-0 flex-1 truncate font-mono text-xs text-foreground-body">
              {str(file, "path", "name") ?? inline(file)}
            </p>
            {size !== null && (
              <span className="shrink-0 text-xs text-foreground-subtle">
                {formatBytes(size)}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}

export function DocsList({ results }: ResultsProps) {
  return (
    <div className={RESULT_GRID}>
      {results.map((doc, i) => {
        const docUrl = str(doc, "doc_url");
        const section = str(doc, "section");
        return (
          <div
            key={resultItemKey(doc, i)}
            className={CARD + " flex items-start gap-2.5 p-2.5"}
          >
            <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-surface-hover">
              <Icon
                icon={BookOpenIcon}
                size={15}
                className="text-foreground-secondary"
              />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-1.5">
                <p className="min-w-0 truncate text-body-sm font-medium leading-inherit text-foreground-emphasis">
                  {str(doc, "title", "path") ?? inline(doc)}
                </p>
                {section && (
                  <span className="shrink-0 truncate text-caption-sm leading-inherit text-foreground-subtle">
                    {section}
                  </span>
                )}
              </div>
              {str(doc, "snippet") && (
                <p className="truncate text-xs text-foreground-tertiary">
                  {str(doc, "snippet")}
                </p>
              )}
            </div>
            {docUrl && (
              <Link
                href={docUrl}
                target="_blank"
                rel="noreferrer"
                aria-label="Open doc"
                className="shrink-0 rounded-full p-1 text-foreground-subtle transition-colors hover:bg-surface-hover hover:text-foreground-body"
              >
                <Icon icon={LinkSquare01Icon} size={14} />
              </Link>
            )}
          </div>
        );
      })}
    </div>
  );
}
