"use client";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";

import {
  formatMaskedValue,
  typeBadgeLabel,
  type CredentialView,
} from "../../helpers";
import {
  CheckmarkSquare02Icon,
  Delete02Icon,
  Loading03Icon,
  SquareIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  credential: CredentialView;
  selected: boolean;
  onToggleSelected: () => void;
  onDelete: () => void;
  isDeleting?: boolean;
}

export function CredentialRow({
  credential,
  selected,
  onToggleSelected,
  onDelete,
  isDeleting = false,
}: Props) {
  return (
    <div
      data-selected={selected}
      className="flex w-full items-center justify-between py-3 pl-3 pr-5 transition-colors data-[selected=true]:bg-surface-hover"
    >
      <div className="flex items-center gap-3">
        {credential.isManaged ? (
          <div className="size-5 shrink-0" aria-hidden="true" />
        ) : (
          <button
            type="button"
            role="checkbox"
            aria-checked={selected}
            aria-label={`Select ${credential.title}`}
            onClick={onToggleSelected}
            className={`shrink-0 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-action ${
              selected
                ? "text-foreground-emphasis hover:text-foreground-strong"
                : "text-foreground-tertiary hover:text-foreground-body"
            }`}
          >
            {selected ? (
              <Icon icon={CheckmarkSquare02Icon} size={20} />
            ) : (
              <Icon icon={SquareIcon} size={20} />
            )}
          </button>
        )}

        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-3">
            <span className="text-body font-medium leading-5.5 text-foreground-base">
              {credential.title}
            </span>
            <span className="inline-flex items-center justify-center rounded-lg bg-surface-hover px-2 py-0.5 text-caption font-medium leading-5 text-foreground-body">
              {credential.isSignIn
                ? "For chats"
                : typeBadgeLabel(credential.type)}
            </span>
          </div>
          <div className="flex items-center gap-3 leading-5">
            <span className="text-caption-sm font-medium uppercase leading-inherit tracking-eyebrow text-foreground-body">
              {credential.blocksNote ?? formatMaskedValue(credential)}
            </span>
          </div>
        </div>
      </div>

      {credential.isManaged ? (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <span
                tabIndex={0}
                className="text-caption-sm font-medium uppercase leading-inherit tracking-eyebrow text-foreground-body focus:outline-none focus-visible:ring-2 focus-visible:ring-action"
              >
                Managed
              </span>
            </TooltipTrigger>
            <TooltipContent side="top">
              Managed by AutoGPT — cannot be removed
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      ) : (
        <button
          type="button"
          onClick={onDelete}
          disabled={isDeleting}
          aria-busy={isDeleting}
          aria-label={`Delete ${credential.title}`}
          className="inline-flex size-5 items-center justify-center text-foreground-base transition-colors hover:text-error focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-brand disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:text-foreground-base"
        >
          {isDeleting ? (
            <Icon icon={Loading03Icon} size={20} className="animate-spin" />
          ) : (
            <Icon icon={Delete02Icon} size={20} />
          )}
        </button>
      )}
    </div>
  );
}
