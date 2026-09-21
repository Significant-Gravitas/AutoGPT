"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { BookOpen01Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { SectionHeader } from "../SectionHeader";
import { SHELF_SIZE } from "../SkillsSection/helpers";
import { useSkillsSection } from "../SkillsSection/useSkillsSection";
import { SkillDialog } from "./components/SkillDialog";
import { SkillRow } from "./components/SkillRow";

const HEADING_ID = "skills-heading";

interface Props {
  category?: string | null;
}

export function SkillsList({ category }: Props) {
  const { skills, total, installedSlugs, isLoading, isError, refetch } =
    useSkillsSection({ category });
  const [openSlug, setOpenSlug] = useState<string | null>(null);

  if (!isLoading && !isError && skills.length === 0) return null;

  return (
    <section
      id="skills"
      aria-labelledby={HEADING_ID}
      className="mb-16 scroll-mt-24"
    >
      <SectionHeader
        size="small"
        titleIcon={<Icon icon={BookOpen01Icon} size={22} aria-hidden />}
        title="Skills"
        titleId={HEADING_ID}
        subtitle="Playbooks your experts pick up as they work."
        action={
          total > SHELF_SIZE
            ? { label: "Browse all skills", href: "/marketplace/skills" }
            : undefined
        }
      />
      {isLoading ? (
        <div
          role="status"
          aria-busy="true"
          aria-label="Loading skills"
          className="grid grid-cols-1 gap-x-8 md:grid-cols-2"
        >
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} className="my-2 h-12 w-full rounded-xl" />
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
      ) : (
        <ul className="grid grid-cols-1 gap-x-8 md:grid-cols-2">
          {skills.map((skill) => (
            <SkillRow
              key={skill.slug}
              skill={skill}
              isInstalled={installedSlugs.has(skill.slug)}
              onSee={() => setOpenSlug(skill.slug)}
            />
          ))}
        </ul>
      )}
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
