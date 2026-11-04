import type { Expert } from "@/app/api/__generated__/models/expert";
import type { ExpertCredentialRef } from "@/app/api/__generated__/models/expertCredentialRef";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { IntegrationLogo } from "@/components/molecules/IntegrationLogo/IntegrationLogo";
import { cn } from "@/lib/utils";
import { formatTimeAgo } from "@/lib/utils/time";
import { taskKey } from "@/app/(platform)/tasks/data/helpers";
import type { Task } from "@/app/(platform)/tasks/data/types";
import {
  statusLabel,
  statusTone,
} from "@/app/(platform)/tasks/components/InboxRow/helpers";
import type { Tone } from "@/components/styles/tones";
import {
  Calendar03Icon,
  CheckListIcon,
  WorkflowSquare01Icon,
} from "@hugeicons/core-free-icons";
import Link from "next/link";
import type { CSSProperties, ReactNode } from "react";
import type { RecipientOption } from "../../../../ChatInput/components/RecipientChip";
import { getAvatarGlow } from "../../../helpers";
import { getWorkInfo, type PanelSchedule } from "../helpers";

const CARD = "rounded-panel bg-surface shadow-card";
const FOCUS =
  "focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none";
interface HeaderProps {
  recipient: RecipientOption;
  role: string;
  description: string | null;
}

export function InfoHeader({ recipient, role, description }: HeaderProps) {
  return (
    <header className="flex flex-col gap-3">
      <span
        className="inline-flex self-start rounded-2xl shadow-(--avatar-glow)"
        style={
          { "--avatar-glow": getAvatarGlow(recipient, 24) } as CSSProperties
        }
      >
        {recipient.id === null ? (
          <AutopilotAvatar size={56} shape="tile" />
        ) : (
          <ExpertAvatar
            name={recipient.name}
            avatarUrl={recipient.avatarUrl}
            size={56}
            shape="tile"
          />
        )}
      </span>
      <div className="flex min-w-0 flex-col gap-1">
        <div className="flex min-w-0 items-center gap-2">
          <Text variant="lead-semibold" as="h2" tone="ink" className="truncate">
            {recipient.name}
          </Text>
          <span className="inline-flex h-5 shrink-0 items-center rounded-control bg-surface-wash px-1.5">
            <Text variant="small" weight="medium" as="span" tone="text">
              {role}
            </Text>
          </span>
        </div>
        {description && (
          <Text variant="inbox" as="p" tone="quiet" className="line-clamp-2">
            {description}
          </Text>
        )}
      </div>
    </header>
  );
}

interface GroupProps {
  label: string;
  count?: number;
  action?: ReactNode;
  children: ReactNode;
}

function InfoGroup({ label, count, action, children }: GroupProps) {
  return (
    <section className="flex flex-col gap-2">
      <div className="flex h-4 items-center justify-between px-1">
        <Text variant="small" weight="medium" as="h3" tone="quiet">
          {label}
          {count !== undefined && count > 0 && (
            <span className="ml-1.5 font-normal text-foreground-faint tabular-nums">
              {count}
            </span>
          )}
        </Text>
        {action}
      </div>
      {children}
    </section>
  );
}

function Tile({ children }: { children: ReactNode }) {
  return (
    <span className="grid size-[26px] shrink-0 place-items-center rounded-md bg-surface-tray text-foreground-text">
      {children}
    </span>
  );
}

function List({ children }: { children: ReactNode }) {
  return <ul className={cn(CARD, "flex flex-col p-1")}>{children}</ul>;
}

interface RowProps {
  tile: ReactNode;
  title: string;
  subtitle?: string;
  trailing?: ReactNode;
  href?: string;
}

function Row({ tile, title, subtitle, trailing, href }: RowProps) {
  const body = (
    <>
      <Tile>{tile}</Tile>
      <span className="flex min-w-0 flex-1 flex-col">
        <Text variant="inbox" as="span" tone="ink" className="truncate">
          {title}
        </Text>
        {subtitle && (
          <Text variant="small" as="span" tone="quiet" className="truncate">
            {subtitle}
          </Text>
        )}
      </span>
      {trailing}
    </>
  );
  const rowClass = cn(
    "flex items-center gap-2.5 rounded-lg px-2",
    subtitle ? "min-h-12 py-1.5" : "h-10",
  );

  return (
    <li className="relative not-first:before:absolute not-first:before:-top-px not-first:before:right-2 not-first:before:left-12 not-first:before:h-px not-first:before:bg-border-hairline">
      {href ? (
        <Link
          href={href}
          className={cn(
            rowClass,
            "transition-colors hover:bg-surface-tray",
            FOCUS,
          )}
        >
          {body}
        </Link>
      ) : (
        <div className={rowClass}>{body}</div>
      )}
    </li>
  );
}

function Pill({
  children,
  tone = "text",
}: {
  children: ReactNode;
  tone?: Tone;
}) {
  return (
    <span className="inline-flex h-5 shrink-0 items-center rounded-control bg-surface-wash px-1.5">
      <Text variant="small" weight="medium" as="span" tone={tone}>
        {children}
      </Text>
    </span>
  );
}

function Meta({ children }: { children: ReactNode }) {
  return (
    <Text variant="small" as="span" tone="quiet" className="shrink-0">
      {children}
    </Text>
  );
}

function Empty({ children }: { children: ReactNode }) {
  return (
    <div className={cn(CARD, "px-3 py-3")}>
      <Text variant="inbox" as="p" tone="quiet">
        {children}
      </Text>
    </div>
  );
}

function Loading() {
  return (
    <div className={cn(CARD, "flex items-center gap-2.5 p-3")}>
      <Skeleton className="size-[26px] rounded-md" />
      <div className="flex flex-1 flex-col gap-1.5">
        <Skeleton className="h-3.5 w-2/3" />
        <Skeleton className="h-3 w-1/3" />
      </div>
    </div>
  );
}

interface SchedulesProps {
  schedules: PanelSchedule[];
  isPaused: boolean;
  isLoading: boolean;
}

export function ScheduledGroup({
  schedules,
  isPaused,
  isLoading,
}: SchedulesProps) {
  return (
    <InfoGroup label="Scheduled" count={schedules.length}>
      {isLoading ? (
        <Loading />
      ) : schedules.length === 0 ? (
        <Empty>Nothing scheduled yet.</Empty>
      ) : (
        <List>
          {schedules.map((schedule) => (
            <Row
              key={schedule.id}
              tile={<Icon icon={Calendar03Icon} size={14} />}
              title={schedule.title}
              subtitle={schedule.cadence}
              trailing={
                isPaused ? (
                  <Pill>Paused</Pill>
                ) : schedule.nextRun ? (
                  <Meta>{schedule.nextRun}</Meta>
                ) : null
              }
            />
          ))}
        </List>
      )}
    </InfoGroup>
  );
}

export function WorkInfoGroup({ expert }: { expert: Expert }) {
  return (
    <InfoGroup label="Work info">
      <List>
        {getWorkInfo(expert).map((row) => (
          <Row
            key={row.label}
            tile={<Icon icon={row.icon} size={14} />}
            title={row.label}
            trailing={
              <Text variant="inbox" as="span" tone="text" className="shrink-0">
                {row.value}
              </Text>
            }
          />
        ))}
      </List>
    </InfoGroup>
  );
}

interface CredentialsProps {
  credentials: ExpertCredentialRef[];
  isLoading: boolean;
}

export function ConnectionsGroup({ credentials, isLoading }: CredentialsProps) {
  return (
    <InfoGroup label="Connections" count={credentials.length}>
      {isLoading ? (
        <Loading />
      ) : credentials.length === 0 ? (
        <Empty>No connections granted yet.</Empty>
      ) : (
        <List>
          {credentials.map((credential) => {
            const service = credential.service_name || credential.provider;
            return (
              <Row
                key={credential.credential_id}
                tile={
                  <IntegrationLogo
                    provider={credential.provider}
                    alt={service}
                    size={16}
                  />
                }
                title={service}
                subtitle={
                  credential.title !== service ? credential.title : undefined
                }
              />
            );
          })}
        </List>
      )}
    </InfoGroup>
  );
}

export function SkillsGroup({ skills }: { skills: string[] }) {
  return (
    <InfoGroup label="Skills" count={skills.length}>
      {skills.length === 0 ? (
        <Empty>No skills listed yet.</Empty>
      ) : (
        <ul className={cn(CARD, "flex flex-wrap gap-1.5 p-2.5")}>
          {skills.map((skill) => (
            <li
              key={skill}
              className="inline-flex h-7 items-center rounded-lg bg-surface-tray px-2.5"
            >
              <Text variant="inbox" as="span" tone="text">
                {skill}
              </Text>
            </li>
          ))}
        </ul>
      )}
    </InfoGroup>
  );
}

export function WorkflowsGroup({
  workflows,
}: {
  workflows: Expert["workflows"];
}) {
  const named = workflows.filter((workflow) => workflow.name);
  return (
    <InfoGroup label="Workflows" count={named.length}>
      {named.length === 0 ? (
        <Empty>No workflows yet.</Empty>
      ) : (
        <List>
          {named.map((workflow) => (
            <Row
              key={workflow.id}
              tile={<Icon icon={WorkflowSquare01Icon} size={14} />}
              title={workflow.name ?? ""}
              subtitle={workflow.description ?? undefined}
              trailing={
                workflow.schedule_id || workflow.schedule_cron ? (
                  <Pill>Scheduled</Pill>
                ) : null
              }
            />
          ))}
        </List>
      )}
    </InfoGroup>
  );
}

interface TasksProps {
  tasks: Task[];
  isLoading: boolean;
  /** The whole team's tasks rather than one expert's. */
  isTeam?: boolean;
}

export function TasksGroup({ tasks, isLoading, isTeam = false }: TasksProps) {
  return (
    <InfoGroup
      label={isTeam ? "Recent tasks" : "Tasks"}
      count={isTeam ? undefined : tasks.length}
      action={
        <Link href="/tasks" className={cn("rounded-md hover:underline", FOCUS)}>
          <Text variant="small" as="span" tone="quiet">
            View all
          </Text>
        </Link>
      }
    >
      {isLoading ? (
        <Loading />
      ) : tasks.length === 0 ? (
        <Empty>{isTeam ? "No tasks yet." : "No tasks assigned yet."}</Empty>
      ) : (
        <List>
          {tasks.map((task) => (
            <Row
              key={task.id}
              href={`/tasks?task=${task.id}`}
              tile={<Icon icon={CheckListIcon} size={14} />}
              title={task.title}
              subtitle={[
                taskKey(task.number),
                isTeam ? (task.assignee?.name ?? "Unassigned") : null,
                formatTimeAgo(task.updated_at),
              ]
                .filter(Boolean)
                .join(" · ")}
              trailing={
                <Pill tone={statusTone(task)}>{statusLabel(task)}</Pill>
              }
            />
          ))}
        </List>
      )}
    </InfoGroup>
  );
}
