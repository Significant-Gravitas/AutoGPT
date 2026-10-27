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
            // TODO(tokens): no exact role for focus-visible:ring-zinc-800 #3E3E43; #3E3E43 is action, the button fill
            className={`shrink-0 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-zinc-800 ${
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
            <span
              // TODO(tokens): no exact role for text-[#1F1F20]
              className="text-body font-medium leading-5.5 text-[#1F1F20]"
            >
              {credential.title}
            </span>
            <span
              // TODO(tokens): no exact role for bg-[#EFF1F4]
              className="inline-flex items-center justify-center rounded-[10px] bg-[#EFF1F4] px-2 py-[2px] text-caption font-medium leading-5 text-foreground-body"
            >
              {credential.isSignIn
                ? "For chats"
                : typeBadgeLabel(credential.type)}
            </span>
          </div>
          <div className="flex items-center gap-3 leading-5">
            <span
              // TODO(tokens): no exact text token for tracking-[1.1px]
              className="text-caption-sm font-medium uppercase leading-inherit tracking-[1.1px] text-foreground-body"
            >
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
                // TODO(tokens): no exact role for focus-visible:ring-zinc-800 #3E3E43; #3E3E43 is action, the button fill; no exact text token for tracking-[1.1px]
                className="text-caption-sm font-medium uppercase leading-inherit tracking-[1.1px] text-foreground-body focus:outline-none focus-visible:ring-2 focus-visible:ring-zinc-800"
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
          // TODO(tokens): no exact role for focus-visible:ring-purple-400 #925cf7, text-[#1F1F20], disabled:hover:text-[#1F1F20]
          className="inline-flex size-5 items-center justify-center text-[#1F1F20] transition-colors hover:text-error focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-purple-400 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:text-[#1F1F20]"
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
