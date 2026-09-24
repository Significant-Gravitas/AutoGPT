"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Book04Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { SectionHeader } from "../SectionHeader";
import {
  SHELF_GRID,
  SHELF_MAX_SIZE,
  SHELF_PREVIEW_SIZE,
} from "../Shelf/helpers";
import { ShelfMoreButton } from "../Shelf/ShelfMoreButton";
import { SkillTopicChips } from "../SkillTopicChips/SkillTopicChips";
import { formatCategoryLabel } from "../SkillsSection/helpers";
import { useSkillsSection } from "../SkillsSection/useSkillsSection";
import { SkillDialog } from "./components/SkillDialog";
import { SkillTile } from "./components/SkillTile";

const HEADING_ID = "skills-heading";

interface Props {
  category?: string | null;
}

export function SkillsList({ category }: Props) {
  // Seeded from the page filter and reset with it by the parent's `key`, so
  // the chips narrow the shelf further without fighting the page.
  const [topic, setTopic] = useState(category ?? null);
  // "Load all" widens the page it asks for rather than slicing a page it
  // already holds: the shelf only ever fetches the tiles it shows.
  const [pageSize, setPageSize] = useState(SHELF_PREVIEW_SIZE);
  const { skills, total, installedSlugs, isLoading, isError, refetch } =
    useSkillsSection({ category: topic, pageSize });
  const [openSlug, setOpenSlug] = useState<string | null>(null);
  const isExpanded = pageSize > SHELF_PREVIEW_SIZE;
  const isEmpty = !isLoading && !isError && skills.length === 0;

  function selectTopic(next: string | null) {
    setTopic(next);
    setPageSize(SHELF_PREVIEW_SIZE);
  }

  // An empty shelf keeps its chips so the reader can step back out of the
  // topic they picked; with no topic of their own there is nothing to step
  // back to, and the section goes.
  if (isEmpty && topic === (category ?? null)) return null;

  return (
    <section
      id="skills"
      aria-labelledby={HEADING_ID}
      className="mb-16 scroll-mt-24"
    >
      <SectionHeader
        size="small"
        titleIcon={<Icon icon={Book04Icon} size="2.2rem" aria-hidden />}
        title="Skills"
        titleId={HEADING_ID}
        subtitle="Playbooks your experts pick up as they work."
        filters={
          <SkillTopicChips
            selected={topic}
            onSelect={selectTopic}
            size="medium"
          />
        }
      />
      {isLoading ? (
        <div
          role="status"
          aria-busy="true"
          aria-label="Loading skills"
          className={SHELF_GRID}
        >
          {Array.from({ length: SHELF_PREVIEW_SIZE }, (_, i) => (
            <Skeleton key={i} className="h-16 w-full rounded-xl" />
          ))}
        </div>
      ) : isError ? (
        <div className="flex items-center gap-2 text-sm text-zinc-600">
          <span>Couldn&apos;t load skills right now.</span>
          <button
            type="button"
            onClick={() => refetch()}
            className="font-medium text-accent underline-offset-2 transition-colors hover:underline"
          >
            Retry
          </button>
        </div>
      ) : isEmpty ? (
        <div className="flex items-center gap-2 text-sm text-zinc-600">
          <span>
            {topic
              ? `No ${formatCategoryLabel(topic).toLowerCase()} skills yet.`
              : "No skills yet."}
          </span>
          {topic ? (
            <button
              type="button"
              onClick={() => selectTopic(null)}
              className="font-medium text-accent underline-offset-2 transition-colors hover:underline"
            >
              Show all
            </button>
          ) : null}
        </div>
      ) : (
        <ul className={SHELF_GRID}>
          {skills.map((skill) => (
            <SkillTile
              key={skill.slug}
              skill={skill}
              isInstalled={installedSlugs.has(skill.slug)}
              onSee={() => setOpenSlug(skill.slug)}
            />
          ))}
        </ul>
      )}
      <div className="mt-6 flex flex-wrap items-center gap-2">
        {total > SHELF_PREVIEW_SIZE ? (
          <ShelfMoreButton
            isExpanded={isExpanded}
            count={Math.min(total, SHELF_MAX_SIZE)}
            isAll={total <= SHELF_MAX_SIZE}
            noun="skills"
            onToggle={() =>
              setPageSize(
                isExpanded
                  ? SHELF_PREVIEW_SIZE
                  : Math.min(total, SHELF_MAX_SIZE),
              )
            }
          />
        ) : null}
        <Button
          as="NextLink"
          href="/marketplace/skills"
          variant="secondary"
          size="small"
        >
          Browse all skills
        </Button>
      </div>
      {openSlug ? (
        <SkillDialog
          key={openSlug}
          slug={openSlug}
          onClose={() => setOpenSlug(null)}
        />
      ) : null}
    </section>
  );
}
