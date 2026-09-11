"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { BookOpen01Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import { SectionHeader } from "../SectionHeader";
import { SkillCard } from "./components/SkillCard";
import { SHELF_SIZE } from "./helpers";
import { useSkillsSection } from "./useSkillsSection";

const HEADING_ID = "skills-heading";

interface Props {
  category?: string | null;
}

export function SkillsSection({ category }: Props) {
  const {
    isLoggedIn,
    skills,
    total,
    installedSlugs,
    isLoading,
    isError,
    refetch,
  } = useSkillsSection({ category });

  // Under a category filter an empty shelf means "no skills in this
  // category", so the whole section goes rather than offering an empty state.
  if (!isLoading && !isError && skills.length === 0 && category) return null;
  // Visitors get nothing when there is nothing to show, as Experts does.
  if (!isLoading && !isError && skills.length === 0 && !isLoggedIn) return null;

  return (
    <section
      id="skills"
      aria-labelledby={HEADING_ID}
      className="mb-20 scroll-mt-24"
    >
      <SectionHeader
        titleIcon={<Icon icon={BookOpen01Icon} size={30} aria-hidden />}
        title="Otto Skills"
        titleId={HEADING_ID}
        subtitle="Playbooks your Otto follows — from brand voice to cold outreach. Install one and it knows how."
        action={sectionAction({ isLoggedIn, total })}
      />
      {isLoggedIn ? (
        <div className="-mt-3 mb-6">
          <Link
            href="/library/skills"
            className="text-sm font-medium text-accent transition-colors hover:text-accent/80"
          >
            …or teach it one of your own
          </Link>
        </div>
      ) : null}

      {isLoading ? (
        <div
          role="status"
          aria-busy="true"
          aria-label="Loading skills"
          className="grid grid-cols-1 gap-5 md:grid-cols-2"
        >
          {[0, 1].map((i) => (
            <Skeleton key={i} className="h-[18rem] w-full rounded-2xl" />
          ))}
        </div>
      ) : isError ? (
        <ShelfError onRetry={refetch} />
      ) : skills.length === 0 ? (
        <EmptyShelf />
      ) : (
        <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
          {skills.map((skill) => (
            <SkillCard
              key={skill.slug}
              skill={skill}
              isInstalled={installedSlugs.has(skill.slug)}
            />
          ))}
        </div>
      )}
    </section>
  );
}

function sectionAction({
  isLoggedIn,
  total,
}: {
  isLoggedIn: boolean;
  total: number;
}) {
  if (total > SHELF_SIZE)
    return { label: "Browse all skills", href: "/marketplace/skills" };
  if (isLoggedIn) return { label: "Your skills", href: "/library/skills" };
  return undefined;
}

// The shelf is secondary on this page, so a failure is one line — the page
// already reserves the full ErrorCard for its own failure.
function ShelfError({ onRetry }: { onRetry: () => void }) {
  return (
    <div className="flex items-center gap-2 text-sm text-zinc-600">
      <span>Couldn&apos;t load skills right now.</span>
      <button
        type="button"
        onClick={onRetry}
        className="font-medium text-accent underline-offset-2 transition-colors hover:underline"
      >
        Retry
      </button>
    </div>
  );
}

function EmptyShelf() {
  return (
    <div
      className="flex flex-col items-center justify-center gap-3 rounded-large border border-dashed border-zinc-200 px-6 py-16 text-center"
      data-testid="skills-shelf-empty"
    >
      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-violet-50">
        <Icon
          icon={BookOpen01Icon}
          size={24}
          className="text-violet-700"
          aria-hidden
        />
      </div>
      <Text variant="h4" className="text-zinc-900">
        Nothing published yet
      </Text>
      <Text variant="body" className="max-w-md !text-zinc-600">
        Skills from the community will show up here. In the meantime you can
        teach your Otto one of your own.
      </Text>
      <Link
        href="/library/skills"
        className="text-sm font-medium text-accent transition-colors hover:text-accent/80"
      >
        Go to your skills →
      </Link>
    </div>
  );
}
