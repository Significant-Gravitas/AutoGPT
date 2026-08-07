import { Expert } from "@/app/api/__generated__/models/expert";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { cn } from "@/lib/utils";
import { getExpertAccent } from "../helpers";
import {
  ArrowRight02Icon,
  CheckmarkCircle02Icon,
  FlashIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
            <AvatarFallback>{expert.name}</AvatarFallback>
          </Avatar>
          <span
            className={cn(
              "inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-sm font-medium",
              accent.pill,
            )}
          >
            <Icon icon={accent.roleIcon} size={14} />
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

        {expert.skills && expert.skills.length > 0 ? (
          <div>
            <div className="mb-1.5 text-[11px] font-medium uppercase tracking-[0.14em] text-zinc-400">
              Skills
            </div>
            <div className="flex flex-wrap gap-1.5">
              {expert.skills.slice(0, 3).map((skill) => (
                <span
                  key={skill}
                  className="rounded-full bg-zinc-50 px-2.5 py-1 text-xs font-medium text-zinc-500 ring-1 ring-inset ring-zinc-200/80"
                >
                  {skill}
                </span>
              ))}
              {expert.skills.length > 3 ? (
                <span className="px-1 py-1 text-xs font-medium text-zinc-400">
                  +{expert.skills.length - 3}
                </span>
              ) : null}
            </div>
          </div>
        ) : null}

        <div className="mt-auto flex items-center justify-between pt-2">
          <span className="flex items-center gap-2 text-base text-zinc-500">
            <Icon icon={FlashIcon} size={18} className={accent.icon} />
            {expert.workflows.length}{" "}
            {expert.workflows.length === 1 ? "workflow" : "workflows"}
          </span>
          {isHired ? (
            <span className="flex items-center gap-1.5 text-base font-medium text-emerald-600">
              <Icon icon={CheckmarkCircle02Icon} size={18} />
              Hired
            </span>
          ) : (
            <span className="flex items-center gap-1.5 text-base font-medium text-zinc-400 transition-colors duration-200 group-hover:text-zinc-900">
              Hire
              <Icon
                icon={ArrowRight02Icon}
                size={16}
                className="transition-transform duration-200 group-hover:translate-x-0.5"
              />
            </span>
          )}
        </div>
      </div>
    </button>
  );
}
