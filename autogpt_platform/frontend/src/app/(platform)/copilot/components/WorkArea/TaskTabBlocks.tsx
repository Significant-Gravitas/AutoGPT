"use client";

import type { Task } from "@/app/api/__generated__/models/task";
import type { TaskArtifact } from "@/app/api/__generated__/models/taskArtifact";
import type { TaskEvidence } from "@/app/api/__generated__/models/taskEvidence";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  CheckmarkCircle02Icon,
  File02Icon,
  HelpCircleIcon,
  Link01Icon,
} from "@hugeicons/core-free-icons";
import {
  InfoGroup,
  InfoList,
  InfoMeta,
} from "../EmptySession/components/ExpertInfoPanel/components/InfoParts";
import { BlockNote, BlockRow } from "./blocks";
import { fileTypeLabel } from "./OverviewTab/helpers";
import { documentOf, hostOf, sourceOf } from "./taskBlocks";

interface ResultProps {
  task: Task;
  artifacts: TaskArtifact[];
  isUnverified: boolean;
  onOpenArtifact: (artifact: TaskArtifact) => void;
}

/** How the task ended: a short summary to read at a glance, and under it
 *  the document that holds the full result. */
export function ResultBlock({
  task,
  artifacts,
  isUnverified,
  onOpenArtifact,
}: ResultProps) {
  const outcome = task.failure_reason || task.result;
  const document = task.failure_reason ? null : documentOf(artifacts);
  if (!outcome && !document) return null;

  return (
    <InfoGroup label={task.failure_reason ? "What went wrong" : "Result"}>
      {outcome ? (
        <BlockNote>
          <Text variant="inbox" tone="text" className="text-pretty">
            {outcome}
          </Text>
          {isUnverified ? (
            <Text as="span" variant="small" tone="waiting">
              Not verified yet
            </Text>
          ) : null}
        </BlockNote>
      ) : null}
      {document ? (
        <InfoList>
          <BlockRow
            tile={<Icon icon={File02Icon} size={14} tone="inherit" />}
            title={document.title}
            subtitle="The full result"
            trailing={<InfoMeta>Open</InfoMeta>}
            onClick={() => onOpenArtifact(document)}
          />
        </InfoList>
      ) : null}
    </InfoGroup>
  );
}

interface ArtifactsProps {
  artifacts: TaskArtifact[];
  onOpenArtifact: (artifact: TaskArtifact) => void;
}

/** What the task made besides its result document. */
export function ArtifactsBlock({ artifacts, onOpenArtifact }: ArtifactsProps) {
  const document = documentOf(artifacts);
  const others = artifacts.filter((artifact) => artifact.id !== document?.id);
  if (others.length === 0) return null;

  return (
    <InfoGroup label="Artifacts" count={others.length}>
      <InfoList>
        {others.map((artifact) => (
          <BlockRow
            key={artifact.id}
            tile={<Icon icon={File02Icon} size={14} tone="inherit" />}
            title={artifact.title}
            trailing={
              <InfoMeta>
                {artifact.is_missing
                  ? "Removed"
                  : fileTypeLabel(artifact.title)}
              </InfoMeta>
            }
            disabled={!artifact.workspace_file_id || artifact.is_missing}
            onClick={() => onOpenArtifact(artifact)}
          />
        ))}
      </InfoList>
    </InfoGroup>
  );
}

interface EvidenceProps {
  evidence: TaskEvidence[];
  artifacts: TaskArtifact[];
  onOpenArtifact: (artifact: TaskArtifact) => void;
}

/** The proof, each row leading to what shows it: the file opens beside the
 *  chat and a page opens in a new tab. */
export function EvidenceBlock({
  evidence,
  artifacts,
  onOpenArtifact,
}: EvidenceProps) {
  if (evidence.length === 0) return null;

  return (
    <InfoGroup label="Evidence" count={evidence.length}>
      <InfoList>
        {evidence.map((item) => {
          const file = sourceOf(item, artifacts);
          const host = hostOf(item.url);
          return (
            <BlockRow
              key={item.id}
              tile={
                <Icon
                  icon={
                    !item.verified
                      ? HelpCircleIcon
                      : host && !file
                        ? Link01Icon
                        : CheckmarkCircle02Icon
                  }
                  size={14}
                  tone={item.verified ? "ok" : "inherit"}
                />
              }
              title={item.claim}
              subtitle={
                file?.title ??
                host ??
                (item.excerpt ? `“${item.excerpt}”` : null)
              }
              trailing={
                <InfoMeta>
                  {item.verified ? "Verified" : "Not verified"}
                </InfoMeta>
              }
              onClick={file ? () => onOpenArtifact(file) : undefined}
              href={!file && host ? (item.url ?? undefined) : undefined}
              external
            />
          );
        })}
      </InfoList>
    </InfoGroup>
  );
}
