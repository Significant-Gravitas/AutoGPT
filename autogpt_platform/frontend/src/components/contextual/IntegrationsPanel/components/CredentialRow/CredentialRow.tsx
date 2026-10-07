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
      className="flex w-full items-center justify-between py-3 pr-5 pl-3 transition-colors data-[selected=true]:bg-zinc-100"
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
            className={`shrink-0 transition-colors focus:outline-hidden focus-visible:ring-2 focus-visible:ring-zinc-800 ${
              selected
                ? "text-zinc-800 hover:text-zinc-900"
                : "text-zinc-500 hover:text-zinc-700"
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
            <span className="text-sm leading-[22px] font-medium text-black">
              {credential.title}
            </span>
            <span className="inline-flex items-center justify-center rounded-[10px] bg-slate-100 px-2 py-0.5 text-xs leading-5 font-medium text-zinc-700">
              {typeBadgeLabel(credential.type)}
            </span>
          </div>
          <div className="flex items-center gap-3 leading-5">
            <span className="text-[11px] font-medium tracking-[1.1px] text-zinc-700 uppercase">
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
                className="text-[11px] font-medium tracking-[1.1px] text-zinc-700 uppercase focus:outline-hidden focus-visible:ring-2 focus-visible:ring-zinc-800"
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
          className="inline-flex size-5 items-center justify-center text-black transition-colors hover:text-red-500 focus-visible:ring-1 focus-visible:ring-purple-400 focus-visible:outline-hidden disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:text-black"
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
