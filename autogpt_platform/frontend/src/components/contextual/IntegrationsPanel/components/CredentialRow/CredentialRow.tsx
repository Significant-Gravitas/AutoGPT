"use client";

import {
  CheckSquareIcon,
  SpinnerIcon,
  SquareIcon,
  TrashIcon,
} from "@phosphor-icons/react";

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
      className="flex w-full items-center justify-between py-3 pl-3 pr-5 transition-colors data-[selected=true]:bg-zinc-100"
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
            className={`shrink-0 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-zinc-800 ${
              selected
                ? "text-zinc-800 hover:text-zinc-900"
                : "text-zinc-500 hover:text-zinc-700"
            }`}
          >
            {selected ? (
              <CheckSquareIcon size={20} weight="fill" />
            ) : (
              <SquareIcon size={20} />
            )}
          </button>
        )}

        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-3">
            <span className="text-[14px] font-medium leading-[22px] text-[#1F1F20]">
              {credential.title}
            </span>
            <span className="inline-flex items-center justify-center rounded-[10px] bg-[#EFF1F4] px-2 py-[2px] text-[12px] font-medium leading-[20px] text-[#505057]">
              {typeBadgeLabel(credential.type)}
            </span>
          </div>
          <div className="flex items-center gap-3 leading-[20px]">
            <span className="text-[11px] font-medium uppercase tracking-[1.1px] text-[#505057]">
              {formatMaskedValue(credential)}
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
                className="text-[11px] font-medium uppercase tracking-[1.1px] text-[#505057] focus:outline-none focus-visible:ring-2 focus-visible:ring-zinc-800"
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
          className="inline-flex size-5 items-center justify-center text-[#1F1F20] transition-colors hover:text-red-500 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-purple-400 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:text-[#1F1F20]"
        >
          {isDeleting ? (
            <SpinnerIcon size={20} className="animate-spin" />
          ) : (
            <TrashIcon size={20} />
          )}
        </button>
      )}
    </div>
  );
}
