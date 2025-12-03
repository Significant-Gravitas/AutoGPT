"use client";

import React from "react";
import { Button } from "@/components/atoms/Button/Button";
import {
  PlayIcon,
  PencilIcon,
  TrashIcon,
  CalendarIcon,
} from "@phosphor-icons/react";

interface Props {
  onEdit?: () => void;
  onDelete?: () => void;
  onRun?: () => void;
  onCreateSchedule?: () => void;
  isRunning?: boolean;
  isDeleting?: boolean;
}

export function TemplateActions({
  onEdit,
  onDelete,
  onRun,
  onCreateSchedule,
  isRunning = false,
  isDeleting = false,
}: Props) {
  return (
    <div className="flex gap-2">
      {onRun && (
        <Button
          variant="primary"
          size="small"
          onClick={onRun}
          disabled={isRunning || isDeleting}
          leftIcon={<PlayIcon size={16} />}
        >
          {isRunning ? "Running..." : "Run Template"}
        </Button>
      )}
      {onEdit && (
        <Button
          variant="secondary"
          size="small"
          onClick={onEdit}
          leftIcon={<PencilIcon size={16} />}
        >
          Edit
        </Button>
      )}
      {onCreateSchedule && (
        <Button
          variant="secondary"
          size="small"
          onClick={onCreateSchedule}
          leftIcon={<CalendarIcon size={16} />}
        >
          Schedule
        </Button>
      )}
      {onDelete && (
        <Button
          variant="destructive"
          size="small"
          onClick={onDelete}
          disabled={isRunning || isDeleting}
          leftIcon={<TrashIcon size={16} />}
        >
          {isDeleting ? "Deleting..." : "Delete"}
        </Button>
      )}
    </div>
  );
}
