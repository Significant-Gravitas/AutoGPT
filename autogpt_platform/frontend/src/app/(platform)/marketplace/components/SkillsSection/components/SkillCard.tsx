import type { MarketplaceSkill } from "@/app/api/__generated__/models/marketplaceSkill";
import { Icon } from "@/components/atoms/Icon/Icon";
import { formatProviderName } from "@/components/contextual/IntegrationsPanel/helpers";
import { Text } from "@/components/atoms/Text/Text";
import { BookOpen01Icon, PlugSocketIcon } from "@hugeicons/core-free-icons";
import Link from "next/link";

interface Props {
  skill: MarketplaceSkill;
}

export function SkillCard({ skill }: Props) {
  return (
    <Link
      href={`/marketplace/skills/${skill.slug}`}
      data-testid="skill-card"
      className="group flex flex-col gap-3 rounded-2xl border border-zinc-200/80 bg-white p-6 shadow-[0_1px_2px_rgba(16,24,40,0.04)] outline-none transition-all duration-200 ease-out hover:-translate-y-0.5 hover:border-zinc-300 hover:shadow-[0_16px_40px_-16px_rgba(16,24,40,0.18)] focus-visible:ring-2 focus-visible:ring-zinc-400"
    >
      <span className="inline-flex h-10 w-10 items-center justify-center rounded-xl bg-violet-50 text-violet-600">
        <Icon icon={BookOpen01Icon} size={20} />
      </span>

      <div>
        <div className="text-lg font-semibold tracking-[-0.01em] text-zinc-900">
          {skill.name}
        </div>
        <p className="mt-1.5 line-clamp-3 text-sm leading-relaxed text-zinc-600">
          {skill.description}
        </p>
      </div>

      <div className="mt-auto flex items-center gap-3 pt-1">
        {skill.required_providers.length > 0 ? (
          <span className="inline-flex items-center gap-1 text-xs text-zinc-500">
            <Icon icon={PlugSocketIcon} size={13} />
            Works with{" "}
            {skill.required_providers.map(formatProviderName).join(", ")}
          </span>
        ) : null}
        <Text variant="small" className="!ml-auto !text-zinc-600">
          {skill.install_count} installed
        </Text>
      </div>
    </Link>
  );
}
