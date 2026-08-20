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
import { CARD, HALF } from "./ResultCards";
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
    <div className="grid gap-1.5 sm:grid-cols-2">
      {results.map((request, i) => (
        <div
          key={resultItemKey(request, i)}
          className={CARD + " flex items-start gap-2.5 p-2.5"}
        >
          <div className="min-w-0 flex-1">
            <p className="truncate text-[13px] font-medium text-zinc-800">
              {str(request, "title") ?? inline(request)}
            </p>
            {str(request, "description") && (
              <p className="truncate text-xs text-zinc-500">
                {str(request, "description")}
              </p>
            )}
          </div>
          {str(request, "identifier") && (
            <span className="shrink-0 rounded-full bg-zinc-100 px-2 py-0.5 text-[11px] text-zinc-500">
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
    <div className="grid gap-1.5 sm:grid-cols-2">
      {schedules.map((schedule, i) => {
        const next = str(schedule, "next_run_time");
        const recurring = !!str(schedule, "cron");
        return (
          <div
            key={resultItemKey(schedule, i)}
            className={CARD + " flex items-center gap-2.5 p-2.5"}
          >
            <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-zinc-100">
              <Icon icon={ClockIcon} size={15} className="text-zinc-600" />
            </div>
            <div className="min-w-0 flex-1">
              <p className="truncate text-[13px] font-medium text-zinc-800">
                {str(schedule, "name", "message") ?? inline(schedule)}
              </p>
              {next && (
                <p className="flex items-center gap-1 text-xs text-zinc-500">
                  {recurring && <Icon icon={RepeatIcon} size={11} />}
                  {formatWhen(next)}
                </p>
              )}
            </div>
            {str(schedule, "kind") && (
              <span className="shrink-0 rounded-full bg-zinc-100 px-2 py-0.5 text-[11px] text-zinc-500">
                {str(schedule, "kind") === "copilot_turn" ? "chat" : "agent"}
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
      <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-zinc-100">
        <Icon icon={ClockIcon} size={15} className="text-zinc-600" />
      </div>
      <div className="min-w-0 flex-1">
        <p className="truncate text-[13px] font-medium text-zinc-800">
          Follow-up scheduled
        </p>
        {next && (
          <p className="flex items-center gap-1 text-xs text-zinc-500">
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
    <div className="grid gap-1.5 sm:grid-cols-2">
      {folders.map((folder, i) => {
        const count =
          typeof folder.agent_count === "number" ? folder.agent_count : null;
        return (
          <div
            key={resultItemKey(folder, i)}
            className={CARD + " flex items-center gap-2.5 p-2.5"}
          >
            <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-zinc-100">
              <Icon icon={FolderIcon} size={15} className="text-zinc-600" />
            </div>
            <p className="min-w-0 flex-1 truncate text-[13px] font-medium text-zinc-800">
              {str(folder, "name") ?? inline(folder)}
            </p>
            {count !== null && (
              <span className="shrink-0 text-xs text-zinc-400">
                {count} agent{count === 1 ? "" : "s"}
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
    <div className={`${CARD} ${HALF} divide-y divide-zinc-100`}>
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
              className="shrink-0 text-zinc-400"
            />
            <p className="min-w-0 flex-1 truncate font-mono text-xs text-zinc-700">
              {str(file, "path", "name") ?? inline(file)}
            </p>
            {size !== null && (
              <span className="shrink-0 text-xs text-zinc-400">
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
    <div className="grid gap-1.5 sm:grid-cols-2">
      {results.map((doc, i) => {
        const docUrl = str(doc, "doc_url");
        const section = str(doc, "section");
        return (
          <div
            key={resultItemKey(doc, i)}
            className={CARD + " flex items-start gap-2.5 p-2.5"}
          >
            <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-zinc-100">
              <Icon icon={BookOpenIcon} size={15} className="text-zinc-600" />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-1.5">
                <p className="min-w-0 truncate text-[13px] font-medium text-zinc-800">
                  {str(doc, "title", "path") ?? inline(doc)}
                </p>
                {section && (
                  <span className="shrink-0 truncate text-[11px] text-zinc-400">
                    {section}
                  </span>
                )}
              </div>
              {str(doc, "snippet") && (
                <p className="truncate text-xs text-zinc-500">
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
                className="shrink-0 rounded-full p-1 text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-zinc-700"
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
