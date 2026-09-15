import type { MarketplaceSkill } from "@/app/api/__generated__/models/marketplaceSkill";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Icon } from "@/components/atoms/Icon/Icon";
import { IntegrationLogo } from "@/components/molecules/IntegrationLogo/IntegrationLogo";
import { formatProviderName } from "@/components/contextual/IntegrationsPanel/helpers";
import { cn } from "@/lib/utils";
import {
  ArrowRight02Icon,
  BookOpen01Icon,
  CheckmarkCircle02Icon,
  Download04Icon,
} from "@hugeicons/core-free-icons";
import Link from "next/link";
import { getCategoryAccent } from "../../ExpertsSection/helpers";
import { formatSkillTitle } from "../helpers";

interface Props {
  skill: MarketplaceSkill;
  isInstalled: boolean;
}

export function SkillCard({ skill, isInstalled }: Props) {
  const { accent, icon } = getCategoryAccent(skill.categories[0]);
  const title = formatSkillTitle(skill.name);
  const providers = skill.required_providers;

  return (
    <Link
      href={`/marketplace/skills/${skill.slug}`}
      data-testid="skill-card"
      className="group relative flex h-full flex-col overflow-hidden rounded-2xl border border-zinc-200/80 bg-white text-left shadow-[0_1px_2px_rgba(16,24,40,0.04)] outline-none transition-[transform,box-shadow,border-color] duration-200 ease-out [touch-action:manipulation] hover:-translate-y-0.5 hover:border-zinc-300 hover:shadow-[0_16px_40px_-16px_rgba(16,24,40,0.18)] focus-visible:ring-2 focus-visible:ring-violet-600 focus-visible:ring-offset-2 motion-reduce:transition-none motion-reduce:hover:transform-none"
    >
      <div
        className={cn(
          "pointer-events-none absolute inset-x-0 top-0 h-28 opacity-60 transition-opacity duration-200 group-hover:opacity-100",
          accent.wash,
        )}
      />
      <div className="relative flex flex-1 flex-col gap-4 p-6">
        <div className="flex items-start justify-between gap-3">
          <span className="inline-flex h-12 w-12 items-center justify-center rounded-xl bg-white ring-1 ring-inset ring-zinc-200/70">
            <Icon
              icon={BookOpen01Icon}
              size={24}
              className={accent.icon}
              aria-hidden
            />
          </span>
          {icon && skill.categories[0] ? (
            <span
              className={cn(
                "inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-sm font-medium",
                accent.pill,
              )}
            >
              <Icon icon={icon} size={14} aria-hidden />
              {formatSkillTitle(skill.categories[0])}
            </span>
          ) : null}
        </div>

        <div className="min-w-0">
          <h3
            title={title}
            className="line-clamp-1 text-lg font-semibold tracking-[-0.01em] text-zinc-900"
          >
            {title}
          </h3>
          <div className="mt-1.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-[13px] text-zinc-500">
            <span className="inline-flex items-center gap-1.5">
              <Avatar className="h-5 w-5">
                {skill.creator_avatar ? (
                  <AvatarImage
                    src={skill.creator_avatar}
                    alt={skill.creator ?? "AutoGPT"}
                  />
                ) : null}
                <AvatarFallback>{skill.creator ?? "AutoGPT"}</AvatarFallback>
              </Avatar>
              by {skill.creator ?? "AutoGPT"}
            </span>
            {providers.length > 0 ? (
              <span className="inline-flex items-center gap-1.5">
                <span aria-hidden>·</span>
                Works with
                <IntegrationLogo provider={providers[0]} size={14} alt="" />
                {formatProviderList(providers)}
              </span>
            ) : null}
          </div>
          <p className="mt-2 line-clamp-3 text-sm leading-relaxed text-zinc-600">
            {skill.description}
          </p>
        </div>

        <div className="mt-auto flex items-center justify-between pt-2">
          <span className="flex items-center gap-2 text-base text-zinc-600">
            <Icon
              icon={Download04Icon}
              size={18}
              className={accent.icon}
              aria-hidden
            />
            {/* Nobody having installed it yet makes a listing new, not failing. */}
            {skill.install_count === 0
              ? "New"
              : `${skill.install_count.toLocaleString()} installed`}
          </span>
          {isInstalled ? (
            <span className="flex items-center gap-1.5 text-base font-medium text-emerald-600">
              <Icon icon={CheckmarkCircle02Icon} size={18} aria-hidden />
              Added
            </span>
          ) : (
            <span className="flex items-center gap-1.5 text-base font-medium text-zinc-500 transition-colors duration-200 group-hover:text-zinc-900">
              View
              <Icon
                icon={ArrowRight02Icon}
                size={16}
                aria-hidden
                className="transition-transform duration-200 group-hover:translate-x-0.5 motion-reduce:transition-none"
              />
            </span>
          )}
        </div>
      </div>
    </Link>
  );
}

function formatProviderList(providers: string[]): string {
  const names = providers.map(formatProviderName).filter(Boolean);
  return new Intl.ListFormat("en", {
    style: "long",
    type: "conjunction",
  }).format(names);
}
