"use client";

import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { ConnectServiceDialog } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/ConnectServiceDialog";
import { formatProviderName } from "@/components/contextual/IntegrationsPanel/helpers";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { IntegrationLogo } from "@/components/molecules/IntegrationLogo/IntegrationLogo";
import { cn } from "@/lib/utils";
import { formatTimeAgo } from "@/lib/utils/time";
import { ArrowLeft02Icon, BookOpen01Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import { notFound } from "next/navigation";
import { getCategoryAccent } from "../../../components/ExpertsSection/helpers";
import { SkillCard } from "../../../components/SkillsSection/components/SkillCard";
import { formatSkillTitle } from "../../../components/SkillsSection/helpers";
import { ExpertSection } from "../../../experts/[expertId]/components/ExpertSection";
import { ConnectStep } from "./ConnectStep";
import { SkillActions } from "./SkillActions";
import { SkillBody } from "./SkillBody";
import { useSkillPage } from "./useSkillPage";

const MAIN_CLASS =
  "mx-auto flex w-full max-w-[760px] flex-col px-6 pb-24 pt-8 md:px-8";

interface Props {
  slug: string;
}

export function SkillPage({ slug }: Props) {
  const {
    skill,
    isLoggedIn,
    isLoading,
    isError,
    isNotFound,
    refetch,
    isReady,
    flagEnabled,
    flagReady,
    isAdded,
    isAdding,
    addToAutoPilot,
    pendingConnections,
    moreSkills,
    isConnectOpen,
    openConnect,
    setIsConnectOpen,
    handleConnected,
  } = useSkillPage(slug);

  // A bookmarked URL reaches this page without passing the marketplace shelf,
  // so the shelf being hidden is not what keeps the feature dark.
  if (flagReady && !flagEnabled) notFound();
  if (isNotFound) notFound();

  if (!flagReady || isLoading) return <SkillPageSkeleton />;

  if (isError || !skill) {
    return (
      <main className={MAIN_CLASS}>
        <BackToMarketplaceLink />
        <ErrorCard
          isSuccess={false}
          responseError={{ message: "Couldn't load this skill right now" }}
          context="marketplace skill"
          onRetry={() => refetch()}
          className="max-w-md"
        />
      </main>
    );
  }

  const title = formatSkillTitle(skill.name);
  const { accent, icon } = getCategoryAccent(skill.categories[0]);
  const providers = skill.required_providers;

  return (
    <main className={MAIN_CLASS}>
      <BackToMarketplaceLink />

      <header>
        <div className="flex flex-wrap items-center gap-4 sm:gap-5">
          <span className="inline-flex h-18 w-18 shrink-0 items-center justify-center rounded-2xl bg-white shadow-sm ring-1 ring-black/5">
            <Icon
              icon={BookOpen01Icon}
              size={32}
              className={accent.icon}
              aria-hidden
            />
          </span>
          <div className="min-w-0 flex-1">
            <h1 className="text-[28px] font-semibold leading-8 tracking-[-0.02em] text-zinc-900">
              {title}
            </h1>
            {icon && skill.categories[0] ? (
              <span
                className={cn(
                  "mt-2 inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-xs font-medium",
                  accent.pill,
                )}
              >
                <Icon icon={icon} size={12} aria-hidden />
                {formatSkillTitle(skill.categories[0])}
              </span>
            ) : null}
          </div>
          <SkillActions
            slug={slug}
            isLoggedIn={isLoggedIn}
            isReady={isReady}
            isAdded={isAdded}
            isAdding={isAdding}
            onAdd={addToAutoPilot}
          />
        </div>

        <p className="mt-5 max-w-[60ch] text-[17px] leading-7 text-zinc-600">
          {skill.description}
        </p>

        <div className="mt-3 flex flex-wrap items-center gap-x-2 gap-y-1 text-[13px] text-zinc-600">
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
          <span aria-hidden>·</span>
          <span>
            {skill.install_count === 0
              ? "New"
              : `${skill.install_count.toLocaleString()} installed`}
          </span>
          {skill.updated_at ? (
            <>
              <span aria-hidden>·</span>
              <span>Updated {formatTimeAgo(String(skill.updated_at))}</span>
            </>
          ) : null}
          {providers.length > 0 ? (
            <>
              <span aria-hidden>·</span>
              <span className="inline-flex items-center gap-1.5">
                Works with
                <IntegrationLogo provider={providers[0]} size={14} alt="" />
                {formatProviderList(providers)}
              </span>
            </>
          ) : null}
        </div>

        {isAdded && pendingConnections.length > 0 ? (
          <ConnectStep names={pendingConnections} onConnect={openConnect} />
        ) : null}
      </header>

      <div className="mt-8 flex flex-col gap-10 border-t border-zinc-200 pt-8">
        <ExpertSection
          title="Instructions"
          description={
            skill.body.trim()
              ? "What your AutoPilot follows once this skill is added."
              : "This skill has no instructions yet."
          }
        >
          {skill.body.trim() ? (
            <SkillBody body={skill.body} title={title} />
          ) : null}
        </ExpertSection>

        {skill.triggers.length > 0 ? (
          <ExpertSection
            title="Triggers"
            count={skill.triggers.length}
            description="Phrases that make your AutoPilot reach for it."
          >
            <div className="flex flex-wrap gap-2">
              {skill.triggers.map((trigger) => (
                <span
                  key={trigger}
                  className="rounded-lg bg-zinc-100 px-2.5 py-1 text-sm font-medium text-zinc-600"
                >
                  {trigger}
                </span>
              ))}
            </div>
          </ExpertSection>
        ) : null}

        {moreSkills.length > 0 ? (
          <ExpertSection title="More skills">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              {moreSkills.map((other) => (
                <SkillCard key={other.slug} skill={other} isInstalled={false} />
              ))}
            </div>
          </ExpertSection>
        ) : null}
      </div>

      <ConnectServiceDialog
        open={isConnectOpen}
        onOpenChange={setIsConnectOpen}
        title={`Connect ${formatProviderList(providers)}`}
        description={`${title}'s steps use ${formatProviderList(providers)}.`}
        onConnected={handleConnected}
      />
    </main>
  );
}

function BackToMarketplaceLink() {
  return (
    <Link
      href="/marketplace#skills"
      className="mb-6 inline-flex w-fit items-center gap-1.5 text-[13px] text-zinc-500 transition-colors hover:text-zinc-900"
      data-testid="skill-back-to-marketplace"
    >
      <Icon icon={ArrowLeft02Icon} size={14} aria-hidden />
      Back to marketplace
    </Link>
  );
}

// Matches the loaded page's shape so nothing shifts when the data lands.
function SkillPageSkeleton() {
  return (
    <main className={MAIN_CLASS} role="status" aria-busy="true">
      <Skeleton className="mb-6 h-4 w-40" />
      <div className="flex flex-wrap items-center gap-4 sm:gap-5">
        <Skeleton className="h-18 w-18 shrink-0 rounded-2xl" />
        <div className="min-w-0 flex-1 space-y-2">
          <Skeleton className="h-7 w-2/3" />
          <Skeleton className="h-5 w-24 rounded-md" />
        </div>
        <Skeleton className="h-9 w-28 rounded-full" />
      </div>
      <Skeleton className="mt-5 h-6 w-4/5" />
      <Skeleton className="mt-3 h-4 w-3/5" />
      <div className="mt-8 space-y-3 border-t border-zinc-200 pt-8">
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-2/3" />
      </div>
    </main>
  );
}

function formatProviderList(providers: string[]): string {
  const names = providers.map(formatProviderName).filter(Boolean);
  return new Intl.ListFormat("en", {
    style: "long",
    type: "conjunction",
  }).format(names);
}
