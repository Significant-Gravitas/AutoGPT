import { Expert } from "@/app/api/__generated__/models/expert";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { cn } from "@/lib/utils";
import {
  ArrowRightIcon,
  CheckCircleIcon,
  LightningIcon,
} from "@phosphor-icons/react";
import { getExpertAccent } from "../helpers";

interface Props {
  expert: Expert;
  isHired: boolean;
  onClick: () => void;
}

export function ExpertCard({ expert, isHired, onClick }: Props) {
  const accent = getExpertAccent(expert.role);

  return (
    <button
      type="button"
      onClick={onClick}
      className="group relative flex flex-col overflow-hidden rounded-2xl border border-zinc-200/80 bg-white text-left shadow-[0_1px_2px_rgba(16,24,40,0.04)] transition-all duration-200 ease-out hover:-translate-y-0.5 hover:border-zinc-300 hover:shadow-[0_16px_40px_-16px_rgba(16,24,40,0.18)]"
    >
      <div
        className={cn(
          "pointer-events-none absolute inset-x-0 top-0 h-28 opacity-60 transition-opacity duration-200 group-hover:opacity-100",
          accent.wash,
        )}
      />
      <div className="relative flex flex-1 flex-col gap-4 p-6">
        <div className="flex items-start justify-between gap-3">
          <Avatar className="h-20 w-20 bg-white shadow-sm ring-1 ring-black/5">
            {expert.avatar_url ? (
              <AvatarImage src={expert.avatar_url} alt={expert.name} />
            ) : null}
            <AvatarFallback>{expert.name.slice(0, 2)}</AvatarFallback>
          </Avatar>
          <span
            className={cn(
              "rounded-full px-3 py-1 text-sm font-medium",
              accent.pill,
            )}
          >
            {expert.role}
          </span>
        </div>

        <div>
          <div className="text-xl font-semibold tracking-[-0.01em] text-zinc-900">
            {expert.name}
          </div>
          {expert.tagline ? (
            <p className="mt-1.5 line-clamp-2 text-base leading-relaxed text-zinc-600">
              {expert.tagline}
            </p>
          ) : null}
        </div>

        <div className="mt-auto flex items-center justify-between pt-2">
          <span className="flex items-center gap-2 text-base text-zinc-500">
            <LightningIcon size={18} weight="fill" className={accent.icon} />
            {expert.workflows.length}{" "}
            {expert.workflows.length === 1 ? "workflow" : "workflows"}
          </span>
          {isHired ? (
            <span className="flex items-center gap-1.5 text-base font-medium text-emerald-600">
              <CheckCircleIcon size={18} weight="fill" />
              Hired
            </span>
          ) : (
            <span className="flex items-center gap-1.5 text-base font-medium text-zinc-400 transition-colors duration-200 group-hover:text-zinc-900">
              Hire
              <ArrowRightIcon
                size={16}
                weight="bold"
                className="transition-transform duration-200 group-hover:translate-x-0.5"
              />
            </span>
          )}
        </div>
      </div>
    </button>
  );
}
