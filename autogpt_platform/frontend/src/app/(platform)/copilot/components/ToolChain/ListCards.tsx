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
import { Text } from "@/components/atoms/Text/Text";
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
            <Text
              variant="body-medium"
              as="p"
              tone="primary"
              unmask={false}
              className="truncate text-[13px]"
            >
              {str(request, "title") ?? inline(request)}
            </Text>
            {str(request, "description") && (
              <Text
                variant="small"
                as="p"
                tone="muted"
                unmask={false}
                className="truncate"
              >
                {str(request, "description")}
              </Text>
            )}
          </div>
          {str(request, "identifier") && (
            <span className="shrink-0 rounded-full bg-zinc-100 px-2 py-0.5 text-[11px] text-muted-foreground">
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
            <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-zinc-100">
              <Icon icon={ClockIcon} size={15} className="text-zinc-600" />
            </div>
            <div className="min-w-0 flex-1">
              <Text
                variant="body-medium"
                as="p"
                tone="primary"
                unmask={false}
                className="truncate text-[13px]"
              >
                {str(schedule, "name", "message") ?? inline(schedule)}
              </Text>
              {next && (
                <Text
                  variant="small"
                  as="p"
                  tone="muted"
                  unmask={false}
                  className="flex items-center gap-1"
                >
                  {recurring && <Icon icon={RepeatIcon} size={11} />}
                  {formatWhen(next)}
                </Text>
              )}
            </div>
            {str(schedule, "kind") && (
              <span className="shrink-0 rounded-full bg-zinc-100 px-2 py-0.5 text-[11px] text-muted-foreground">
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
        <Text
          variant="body-medium"
          as="p"
          tone="primary"
          className="truncate text-[13px]"
        >
          Follow-up scheduled
        </Text>
        {next && (
          <Text
            variant="small"
            as="p"
            tone="muted"
            unmask={false}
            className="flex items-center gap-1"
          >
            {output.is_recurring === true && (
              <Icon icon={RepeatIcon} size={11} />
            )}
            {formatWhen(next)}
          </Text>
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
            <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-zinc-100">
              <Icon icon={FolderIcon} size={15} className="text-zinc-600" />
            </div>
            <Text
              variant="body-medium"
              as="p"
              tone="primary"
              unmask={false}
              className="min-w-0 flex-1 truncate text-[13px]"
            >
              {str(folder, "name") ?? inline(folder)}
            </Text>
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
            <Text
              variant="small"
              as="p"
              tone="secondary"
              unmask={false}
              className="min-w-0 flex-1 truncate font-mono"
            >
              {str(file, "path", "name") ?? inline(file)}
            </Text>
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
    <div className={RESULT_GRID}>
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
                <Text
                  variant="body-medium"
                  as="p"
                  tone="primary"
                  unmask={false}
                  className="min-w-0 truncate text-[13px]"
                >
                  {str(doc, "title", "path") ?? inline(doc)}
                </Text>
                {section && (
                  <span className="shrink-0 truncate text-[11px] text-zinc-400">
                    {section}
                  </span>
                )}
              </div>
              {str(doc, "snippet") && (
                <Text
                  variant="small"
                  as="p"
                  tone="muted"
                  unmask={false}
                  className="truncate"
                >
                  {str(doc, "snippet")}
                </Text>
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
