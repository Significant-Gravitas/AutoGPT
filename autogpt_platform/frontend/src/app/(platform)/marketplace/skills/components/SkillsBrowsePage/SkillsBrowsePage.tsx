"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { InfiniteList } from "@/components/molecules/InfiniteList/InfiniteList";
import { useStoreCategories } from "@/hooks/useStoreCategories";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { ArrowLeft02Icon, BookOpen01Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import { notFound } from "next/navigation";
import { CategoryFilter } from "../../../components/CategoryFilter/CategoryFilter";
import { SearchBar } from "../../../components/SearchBar/SearchBar";
import { SkillCard } from "../../../components/SkillsSection/components/SkillCard";
import { useSkillsBrowsePage } from "./useSkillsBrowsePage";

const GRID_CLASS = "grid grid-cols-1 gap-5 md:grid-cols-2";

export function SkillsBrowsePage() {
  const { enabled, ready } = useFlagStatus(Flag.SKILLS_HUB);
  const {
    search,
    category,
    skills,
    total,
    installedSlugs,
    isLoading,
    isError,
    refetch,
    hasMore,
    isFetchingMore,
    loadMore,
    setSearch,
    setCategory,
  } = useSkillsBrowsePage();
  const { categories } = useStoreCategories();

  if (ready && !enabled) notFound();

  return (
    <main className="mx-auto w-full max-w-[1360px] px-6 pb-16 pt-8 md:px-10 lg:px-14">
      <Link
        href="/marketplace#skills"
        className="mb-6 inline-flex w-fit items-center gap-1.5 text-[13px] text-zinc-500 transition-colors hover:text-zinc-900"
      >
        <Icon icon={ArrowLeft02Icon} size={14} aria-hidden />
        Back to marketplace
      </Link>

      <div className="mb-7 flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
        <div>
          <h1 className="flex items-center gap-2.5 text-3xl font-semibold tracking-[-0.02em] text-zinc-900">
            <Icon icon={BookOpen01Icon} size={30} aria-hidden />
            Skills
          </h1>
          <p className="mt-2 text-base text-zinc-500">
            Playbooks your experts follow. Add a playbook to your library, ready
            to assign to your experts.
          </p>
        </div>
        <SearchBar
          placeholder="Search skills"
          height="h-[2.75rem]"
          width="w-full md:w-[439px]"
          defaultValue={search}
          onSubmit={setSearch}
        />
      </div>

      <CategoryFilter selected={category} onSelect={setCategory} />

      {!ready || isLoading ? (
        <div
          role="status"
          aria-busy="true"
          aria-label="Loading skills"
          className={GRID_CLASS}
        >
          {[0, 1, 2, 3, 4, 5, 6, 7].map((i) => (
            <Skeleton key={i} className="h-[18rem] w-full rounded-2xl" />
          ))}
        </div>
      ) : isError ? (
        <ErrorCard
          isSuccess={false}
          responseError={{ message: "Couldn't load skills right now" }}
          context="marketplace skills"
          onRetry={() => refetch()}
          className="max-w-md"
        />
      ) : skills.length === 0 ? (
        <EmptyResult
          category={category}
          categoryLabel={
            categories.find((entry) => entry.value === category)?.label
          }
          onShowAll={() => setCategory(null)}
        />
      ) : (
        <>
          <p className="mb-4 text-sm text-zinc-500">
            {total.toLocaleString()} {total === 1 ? "skill" : "skills"}
          </p>
          <InfiniteList
            items={skills}
            className={GRID_CLASS}
            hasMore={hasMore}
            isFetchingMore={isFetchingMore}
            onEndReached={loadMore}
            renderItem={(skill) => (
              <SkillCard
                key={skill.slug}
                skill={skill}
                isInstalled={installedSlugs.has(skill.slug)}
              />
            )}
          />
        </>
      )}
    </main>
  );
}

function EmptyResult({
  category,
  categoryLabel,
  onShowAll,
}: {
  category: string | null;
  categoryLabel: string | undefined;
  onShowAll: () => void;
}) {
  return (
    <div
      className="flex flex-col items-center justify-center gap-3 rounded-large border border-dashed border-zinc-200 px-6 py-16 text-center"
      data-testid="skills-browse-empty"
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
        {category
          ? `No ${categoryLabel ?? category} skills yet`
          : "Nothing published yet"}
      </Text>
      {category ? (
        <button
          type="button"
          onClick={onShowAll}
          className="text-sm font-medium text-accent transition-colors hover:text-accent/80"
        >
          Show all
        </button>
      ) : (
        <Link
          href="/library/skills"
          className="text-sm font-medium text-accent transition-colors hover:text-accent/80"
        >
          Go to your skills →
        </Link>
      )}
    </div>
  );
}
