"use client";

import {
  ArrowDown01Icon,
  ArrowUpRight01Icon,
  Calendar03Icon,
  Flag02Icon,
} from "@hugeicons/core-free-icons";
import { Fragment } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import {
  PRIORITY_LABELS,
  STATUS_LABELS,
  taskKey,
} from "../../../../data/helpers";
import type {
  ExpertOption,
  TaskDetail,
  TaskPriority,
  TaskStatus,
  UpdateTaskInput,
} from "../../../../data/types";
import {
  STATUS_ICONS,
  formatDueDate,
  getAssigneeChoices,
  getAssigneeLabel,
  isTaskOverdue,
} from "../../helpers";
import { AssigneeAvatar } from "../AssigneeAvatar/AssigneeAvatar";

interface Props {
  task: TaskDetail;
  experts: ExpertOption[];
  onUpdate: (input: UpdateTaskInput) => void;
}

const STATUS_ORDER: TaskStatus[] = [
  "backlog",
  "todo",
  "in_progress",
  "blocked",
  "in_review",
  "done",
  "cancelled",
];

const PRIORITY_ORDER: TaskPriority[] = [
  "urgent",
  "high",
  "medium",
  "low",
  "none",
];

export function TaskHeader({ task, experts, onUpdate }: Props) {
  const status = STATUS_ICONS[task.status];
  const choices = getAssigneeChoices(experts);
  const isOverdue = isTaskOverdue(task.due_date, task.status);

  return (
    <div className="flex flex-col gap-2">
      <Text variant="small-mono" tone="faint">
        {taskKey(task.number)}
      </Text>
      <Text variant="large-medium" as="h2" tone="ink" className="text-pretty">
        {task.title}
      </Text>
      <div className="flex flex-wrap items-center gap-1.5 pt-1">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="secondary"
              size="xs"
              aria-label="Change status"
              leftIcon={<Icon icon={status.icon} tone={status.tone} />}
              rightIcon={<Icon icon={ArrowDown01Icon} tone="faint" />}
            >
              {STATUS_LABELS[task.status]}
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start">
            {STATUS_ORDER.map((value) => (
              <DropdownMenuItem
                key={value}
                onSelect={() => onUpdate({ status: value })}
              >
                <span className="flex items-center gap-2">
                  <Icon
                    icon={STATUS_ICONS[value].icon}
                    tone={STATUS_ICONS[value].tone}
                  />
                  {STATUS_LABELS[value]}
                </span>
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="secondary"
              size="xs"
              aria-label="Change assignee"
              leftIcon={<AssigneeAvatar assignee={task.assignee} size={16} />}
            >
              {getAssigneeLabel(task.assignee)}
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start">
            {choices.map((choice, index) => (
              <Fragment key={choice.key}>
                {index === 2 ? <DropdownMenuSeparator /> : null}
                <DropdownMenuItem
                  onSelect={() => onUpdate({ assignee: choice.value })}
                >
                  <span className="flex items-center gap-2">
                    <AssigneeAvatar
                      assignee={{
                        type: choice.value.type,
                        id: choice.value.id,
                        name: choice.label,
                        avatar_url: experts.find((e) => e.id === choice.key)
                          ?.avatar_url,
                      }}
                      size={16}
                    />
                    {choice.label}
                  </span>
                </DropdownMenuItem>
              </Fragment>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="secondary"
              size="xs"
              aria-label="Change priority"
              leftIcon={
                <Icon
                  icon={Flag02Icon}
                  tone={task.priority === "urgent" ? "waiting" : "quiet"}
                />
              }
            >
              {PRIORITY_LABELS[task.priority]}
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start">
            {PRIORITY_ORDER.map((value) => (
              <DropdownMenuItem
                key={value}
                onSelect={() => onUpdate({ priority: value })}
              >
                {PRIORITY_LABELS[value]}
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>

        {task.due_date ? (
          <span className="inline-flex h-7 items-center gap-1.5 rounded-control bg-surface-wash px-2.5">
            <Icon
              icon={Calendar03Icon}
              tone={isOverdue ? "waiting" : "quiet"}
            />
            <Text
              variant="small"
              weight="medium"
              tone={isOverdue ? "waiting" : "text"}
            >
              {isOverdue ? "Overdue · " : "Due "}
              {formatDueDate(task.due_date)}
            </Text>
          </span>
        ) : null}

        {task.thread_session_id ? (
          <Button
            as="NextLink"
            href={`/copilot?sessionId=${encodeURIComponent(task.thread_session_id)}`}
            variant="ghost-muted"
            size="xs"
            rightIcon={<Icon icon={ArrowUpRight01Icon} />}
          >
            Open work thread
          </Button>
        ) : null}
      </div>
    </div>
  );
}
