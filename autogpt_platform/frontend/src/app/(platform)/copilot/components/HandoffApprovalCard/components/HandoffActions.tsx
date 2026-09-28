"use client";

import {
  PencilEdit02Icon,
  Shield01Icon,
  Tick02Icon,
  UserSwitchIcon,
} from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import type { CardStatus } from "../../ApprovalQueue/components/ApprovalCard/ApprovalCard";

interface ExpertChoice {
  id: string;
  name: string;
  role: string | null;
}

interface Props {
  status: CardStatus;
  expertName: string;
  canEdit: boolean;
  isEditing: boolean;
  canAlwaysAllow: boolean;
  otherExperts: ExpertChoice[];
  onApprove: () => void;
  onReject: () => void;
  onToggleEdit: () => void;
  onPickExpert: (id: string) => void;
  onAlwaysAllow: () => void;
}

export function HandoffActions({
  status,
  expertName,
  canEdit,
  isEditing,
  canAlwaysAllow,
  otherExperts,
  onApprove,
  onReject,
  onToggleEdit,
  onPickExpert,
  onAlwaysAllow,
}: Props) {
  const busy = status !== "idle";
  return (
    <div className="flex flex-col gap-3 border-t border-zinc-100 px-4 py-3 sm:flex-row sm:items-center">
      <div className="grid grid-cols-2 gap-2 sm:flex sm:flex-wrap sm:items-center">
        <Button
          size="small"
          variant="primary"
          leadingIcon={Tick02Icon}
          loading={status === "approving"}
          disabled={busy}
          onClick={onApprove}
          className="min-w-0"
        >
          {status === "approving" ? "Approving…" : "Approve"}
        </Button>
        <Button
          size="small"
          variant="secondary"
          loading={status === "rejecting"}
          disabled={busy}
          onClick={onReject}
          className="min-w-0"
        >
          {status === "rejecting" ? "Rejecting…" : "Reject"}
        </Button>
        {canEdit && (
          <Button
            size="small"
            variant="ghost"
            leadingIcon={PencilEdit02Icon}
            aria-pressed={isEditing}
            disabled={busy}
            onClick={onToggleEdit}
            className="min-w-0"
          >
            {isEditing ? "Keep brief" : "Edit brief"}
          </Button>
        )}
        {canEdit && otherExperts.length > 0 && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                size="small"
                variant="ghost"
                leadingIcon={UserSwitchIcon}
                disabled={busy}
                className="min-w-0"
              >
                Pick another expert
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" className="w-64">
              {otherExperts.map((expert) => (
                <DropdownMenuItem
                  key={expert.id}
                  onSelect={() => onPickExpert(expert.id)}
                  className="flex flex-col items-start gap-0.5 py-2"
                >
                  <span className="text-sm text-zinc-900">{expert.name}</span>
                  {expert.role && (
                    <span className="text-xs text-zinc-500">{expert.role}</span>
                  )}
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        )}
      </div>
      <span className="hidden flex-1 sm:block" />
      {canAlwaysAllow && (
        <button
          type="button"
          disabled={busy}
          onClick={onAlwaysAllow}
          className="flex items-center gap-1.5 text-xs leading-[18px] text-zinc-600 underline underline-offset-2 hover:text-zinc-900 disabled:opacity-50"
        >
          <Icon icon={Shield01Icon} size={14} />
          Always allow hand-offs to {expertName}
        </button>
      )}
    </div>
  );
}
