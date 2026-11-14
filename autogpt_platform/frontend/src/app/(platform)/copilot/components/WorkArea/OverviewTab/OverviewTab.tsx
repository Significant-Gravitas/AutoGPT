"use client";

import { StateIcon } from "@/app/(platform)/tasks/components/StateIcon/StateIcon";
import { ownerOf } from "@/app/(platform)/tasks/helpers";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { integrationIconSrc } from "@/components/molecules/IntegrationLogo/helpers";
import {
  Attachment01Icon,
  Brain02Icon,
  CommandLineIcon,
  ComputerIcon,
  File02Icon,
  Globe02Icon,
  Loading03Icon,
  MessageQuestionIcon,
  PlugIcon,
  SparklesIcon,
  Tick02Icon,
  UserGroupIcon,
} from "@hugeicons/core-free-icons";
import {
  InfoGroup,
  InfoList,
  InfoMeta,
} from "../../EmptySession/components/ExpertInfoPanel/components/InfoParts";
import { BlockNote, BlockRow, NeedsYouTag } from "../blocks";
import { fileTypeLabel, STEPS_SHOWN } from "./helpers";
import { useChatOverview } from "./useChatOverview";

type OpenTarget = "files" | "computer" | "integrations";

interface Props {
  sessionId: string;
  /** The chat is with an expert, whose integrations can be opened. */
  hasExpert: boolean;
  onOpen: (what: OpenTarget) => void;
  onOpenTask: (taskID: string, title: string) => void;
}

/** What this chat has going on, as blocks: what waits on the user, the
 *  tasks it started, the steps of the turn, what it made, what it used and
 *  what was decided. A block with nothing in it is left out; a chat with
 *  nothing at all gets one line saying what will show up here. */
export function OverviewTab({
  sessionId,
  hasExpert,
  onOpen,
  onOpenTask,
}: Props) {
  const chat = useChatOverview(sessionId);
  const earlier = Math.max(0, chat.steps.length - STEPS_SHOWN);
  const steps = chat.steps.slice(-STEPS_SHOWN);
  const question = chat.pending?.questions[0]?.question ?? null;
  const usedRows = usageRows(chat, hasExpert, onOpen);
  const isEmpty =
    !question &&
    chat.tasks.length === 0 &&
    chat.steps.length === 0 &&
    chat.outputs.length === 0 &&
    usedRows.length === 0 &&
    chat.decisions.length === 0;

  if (isEmpty) {
    return (
      <BlockNote>
        <Text variant="inbox-medium" tone="ink">
          Nothing here yet
        </Text>
        <Text variant="inbox" tone="quiet" className="text-pretty">
          Tasks Otto starts, files it makes and the tools it uses in this chat
          will show up here.
        </Text>
      </BlockNote>
    );
  }

  return (
    <div className="flex flex-col gap-5">
      {question ? (
        <InfoGroup label="Needs you">
          <BlockNote>
            <NeedsYouTag>Waiting on you</NeedsYouTag>
            <Text variant="inbox" tone="ink" className="text-pretty">
              {question}
            </Text>
            <Text variant="small" tone="quiet">
              Answer in the card above the message box.
            </Text>
          </BlockNote>
        </InfoGroup>
      ) : null}

      {chat.tasks.length > 0 ? (
        <InfoGroup label="Tasks" count={chat.tasks.length}>
          <InfoList>
            {chat.tasks.map((task) => (
              <BlockRow
                key={task.id}
                tile={<StateIcon status={task.status} />}
                title={task.title}
                subtitle={task.status_text ?? ownerOf(task, chat.experts).name}
                onClick={() => onOpenTask(task.id, task.title)}
                trailing={
                  (task.open_ask_count ?? 0) > 0 ? (
                    <NeedsYouTag />
                  ) : (task.subtask_count ?? 0) > 0 ? (
                    <InfoMeta>
                      {task.subtasks_done ?? 0}/{task.subtask_count}
                    </InfoMeta>
                  ) : null
                }
              />
            ))}
          </InfoList>
        </InfoGroup>
      ) : null}

      {steps.length > 0 ? (
        <InfoGroup
          label="This turn"
          action={
            earlier > 0 ? (
              <InfoMeta>
                {earlier} earlier {earlier === 1 ? "step" : "steps"}
              </InfoMeta>
            ) : null
          }
        >
          <InfoList>
            {steps.map((step) => (
              <BlockRow
                key={step.key}
                tile={
                  <Icon
                    icon={step.state === "now" ? Loading03Icon : Tick02Icon}
                    size={14}
                    tone={step.state === "done" ? "ok" : "inherit"}
                    spin={step.state === "now"}
                  />
                }
                title={step.text}
              />
            ))}
          </InfoList>
        </InfoGroup>
      ) : null}

      {chat.outputs.length > 0 ? (
        <InfoGroup label="Outputs" count={chat.outputs.length}>
          {chat.preview ? (
            <button
              type="button"
              onClick={() => chat.preview && chat.openFile(chat.preview.file)}
              className="flex flex-col gap-1.5 rounded-panel bg-surface p-4 text-left shadow-(--inbox-shadow-card) transition-shadow duration-150 ease-out outline-none hover:shadow-(--inbox-shadow-card-hover) focus-visible:ring-2 focus-visible:ring-focus-inbox"
            >
              <Text
                as="span"
                variant="body-medium"
                tone="ink"
                className="text-pretty"
              >
                {chat.preview.title}
              </Text>
              {chat.preview.lines.map((line) => (
                <Text
                  key={line}
                  as="span"
                  variant="inbox"
                  tone="text"
                  className="line-clamp-2 text-pretty"
                >
                  {line}
                </Text>
              ))}
              <span aria-hidden className="mt-1 flex flex-col gap-1.5">
                <span className="h-1 w-full rounded-full bg-surface-wash" />
                <span className="h-1 w-4/5 rounded-full bg-surface-wash" />
                <span className="h-1 w-3/5 rounded-full bg-surface-wash" />
              </span>
            </button>
          ) : null}
          <InfoList>
            {chat.outputs.map((file) => (
              <BlockRow
                key={file.id}
                tile={<Icon icon={File02Icon} size={14} tone="inherit" />}
                title={file.name}
                trailing={<InfoMeta>{fileTypeLabel(file.name)}</InfoMeta>}
                onClick={() => chat.openFile(file)}
              />
            ))}
          </InfoList>
        </InfoGroup>
      ) : null}

      {usedRows.length > 0 ? (
        <InfoGroup label="Used in this chat">
          <InfoList>{usedRows}</InfoList>
        </InfoGroup>
      ) : null}

      {chat.decisions.length > 0 ? (
        <InfoGroup label="What you decided" count={chat.decisions.length}>
          <InfoList>
            {chat.decisions.map((decision) => (
              <BlockRow
                key={decision.question}
                tile={
                  <Icon icon={MessageQuestionIcon} size={14} tone="inherit" />
                }
                title={decision.answer}
                subtitle={decision.question}
                wrap
              />
            ))}
          </InfoList>
        </InfoGroup>
      ) : null}
    </div>
  );
}

type Chat = ReturnType<typeof useChatOverview>;

// One row per thing the chat used, and only for what it did use.
function usageRows(
  chat: Chat,
  hasExpert: boolean,
  onOpen: (what: OpenTarget) => void,
) {
  const { usage } = chat;
  const rows = [];
  if (chat.uploads.length > 0) {
    rows.push(
      <BlockRow
        key="uploads"
        tile={<Icon icon={Attachment01Icon} size={14} tone="inherit" />}
        title="Uploads"
        trailing={
          <InfoMeta>
            {chat.uploads[0].name}
            {chat.uploads.length > 1 ? ` +${chat.uploads.length - 1}` : ""}
          </InfoMeta>
        }
        onClick={() => onOpen("files")}
      />,
    );
  }
  if (usage.connectors.length > 0) {
    rows.push(
      <BlockRow
        key="connectors"
        tile={<Icon icon={PlugIcon} size={14} tone="inherit" />}
        title="Connectors"
        onClick={hasExpert ? () => onOpen("integrations") : undefined}
        trailing={
          <span className="flex shrink-0 items-center gap-1.5">
            {usage.connectors.map((provider) => {
              const src = integrationIconSrc(provider);
              return src ? (
                // eslint-disable-next-line @next/next/no-img-element -- a 16px logo from /public, not worth the image pipeline
                <img
                  key={provider}
                  src={src}
                  alt={provider}
                  title={provider}
                  className="size-4 rounded-sm"
                />
              ) : (
                <InfoMeta key={provider}>{provider}</InfoMeta>
              );
            })}
          </span>
        }
      />,
    );
  }
  if (usage.searches + usage.pagesRead > 0) {
    rows.push(
      <BlockRow
        key="web"
        tile={<Icon icon={Globe02Icon} size={14} tone="inherit" />}
        title="Web search"
        trailing={
          <InfoMeta>
            {[
              usage.searches
                ? `${usage.searches} ${usage.searches === 1 ? "search" : "searches"}`
                : null,
              usage.pagesRead
                ? `${usage.pagesRead} ${usage.pagesRead === 1 ? "page" : "pages"}`
                : null,
            ]
              .filter(Boolean)
              .join(" · ")}
          </InfoMeta>
        }
      />,
    );
  }
  if (usage.commands > 0) {
    rows.push(
      <BlockRow
        key="commands"
        tile={<Icon icon={CommandLineIcon} size={14} tone="inherit" />}
        title="Commands"
        trailing={<InfoMeta>{usage.commands} run</InfoMeta>}
      />,
    );
  }
  if (usage.skills.length > 0) {
    rows.push(
      <BlockRow
        key="skills"
        tile={<Icon icon={SparklesIcon} size={14} tone="inherit" />}
        title="Skills"
        trailing={<InfoMeta>{usage.skills.join(", ")}</InfoMeta>}
      />,
    );
  }
  if (chat.teammates.length > 0) {
    rows.push(
      <BlockRow
        key="teammates"
        tile={<Icon icon={UserGroupIcon} size={14} tone="inherit" />}
        title="Teammates"
        trailing={<InfoMeta>{chat.teammates.join(", ")}</InfoMeta>}
      />,
    );
  }
  if (chat.hasComputer) {
    rows.push(
      <BlockRow
        key="computer"
        tile={<Icon icon={ComputerIcon} size={14} tone="inherit" />}
        title="Computer"
        trailing={<InfoMeta>Open</InfoMeta>}
        onClick={() => onOpen("computer")}
      />,
    );
  }
  if (usage.memoryChecks + usage.notesSaved > 0) {
    rows.push(
      <BlockRow
        key="memory"
        tile={<Icon icon={Brain02Icon} size={14} tone="inherit" />}
        title="Memory"
        trailing={
          <InfoMeta>
            {usage.notesSaved > 0
              ? `${usage.notesSaved} ${usage.notesSaved === 1 ? "note" : "notes"} saved`
              : "Checked"}
          </InfoMeta>
        }
      />,
    );
  }
  return rows;
}
