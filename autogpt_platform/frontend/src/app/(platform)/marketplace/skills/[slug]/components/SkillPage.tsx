"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import {
  ArrowLeft02Icon,
  BookOpen01Icon,
  CheckmarkBadge01Icon,
} from "@hugeicons/core-free-icons";
import Link from "next/link";
import { InstallSkillPanel } from "./InstallSkillPanel/InstallSkillPanel";
import { SkillBody } from "./SkillBody";
import { useSkillPage } from "./useSkillPage";

interface Props {
  slug: string;
}

export function SkillPage({ slug }: Props) {
  const { skill, isLoading, isError } = useSkillPage(slug);

  if (isLoading) {
    return (
      <main className="container max-w-4xl space-y-6 pb-20 pt-16">
        <Skeleton className="h-10 w-2/3" />
        <Skeleton className="h-64 w-full rounded-2xl" />
      </main>
    );
  }

  if (isError || !skill) {
    return (
      <main className="container max-w-4xl pb-20 pt-16">
        <ErrorCard
          isSuccess={false}
          responseError={{ message: "This skill is no longer available" }}
          context="marketplace skill"
        />
      </main>
    );
  }

  return (
    <main className="container max-w-4xl space-y-8 pb-20 pt-16">
      <Link
        href="/marketplace#skills"
        className="inline-flex items-center gap-1 text-sm text-zinc-500 hover:text-zinc-800"
        data-testid="skill-back-to-marketplace"
      >
        <Icon icon={ArrowLeft02Icon} size={14} />
        Back to Marketplace
      </Link>

      <header className="flex flex-col gap-4">
        <div className="flex items-start gap-4">
          <span className="inline-flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-violet-50 text-violet-600">
            <Icon icon={BookOpen01Icon} size={24} />
          </span>
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-3">
              <Text variant="h2">{skill.name}</Text>
              {skill.is_verified ? (
                <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2.5 py-1 text-xs font-medium text-emerald-700">
                  <Icon icon={CheckmarkBadge01Icon} size={13} />
                  Verified
                </span>
              ) : null}
            </div>
            <Text variant="body" className="!mt-2 !text-zinc-600">
              {skill.description}
            </Text>
            <Text variant="small" className="!mt-2 !text-zinc-400">
              {skill.creator ? `By ${skill.creator} · ` : ""}
              {skill.install_count} installed
            </Text>
          </div>
        </div>

        <InstallSkillPanel
          slug={skill.slug}
          requiredProviders={skill.required_providers}
        />
      </header>

      <section className="rounded-2xl border border-zinc-200/80 bg-white p-6">
        <Text variant="large-medium" className="!mb-3">
          What your AutoPilot will follow
        </Text>
        <SkillBody body={skill.body} />
      </section>
    </main>
  );
}
