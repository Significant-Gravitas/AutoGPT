"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { SearchInput } from "@/components/molecules/SearchInput/SearchInput";
import { PlusSignIcon } from "@hugeicons/core-free-icons";
import { AddSkillDialog } from "./AddSkillDialog";
import { ExpertSkillListItem } from "./ExpertSkillListItem";
import { useExpertSkills } from "./useExpertSkills";

interface Props {
  expert: Expert;
  accentClassName?: string;
}

export function ExpertSkillsSection({ expert, accentClassName }: Props) {
  const {
    query,
    setQuery,
    visible,
    hasAny,
    available,
    isLibraryLoading,
    isAddOpen,
    openAdd,
    closeAdd,
    source,
    setSource,
    marketQuery,
    setMarketQuery,
    marketplaceSkills,
    isMarketplaceLoading,
    addSkill,
    addMarketplaceSkill,
    removeSkill,
    isSaving,
  } = useExpertSkills(expert);

  return (
    <section>
      <div className="mb-2.5 flex flex-wrap items-center justify-between gap-3">
        <Text variant="body-medium" tone="primary">
          {expert.name}&apos;s Skills
        </Text>
        <div className="flex items-center gap-2">
          <Button
            variant="secondary"
            size="xs"
            leadingIcon={PlusSignIcon}
            onClick={openAdd}
          >
            Add skill
          </Button>
          <SearchInput
            size="xsmall"
            value={query}
            onChange={setQuery}
            placeholder="Search skills"
            className="w-48"
          />
        </div>
      </div>
      {!hasAny ? (
        <Text variant="body" tone="muted" className="pt-4">
          No skills yet. Add skills from your library so {expert.name} knows how
          you like things done.
        </Text>
      ) : visible.length === 0 ? (
        <Text variant="body" tone="muted" className="pt-4">
          No skills match.
        </Text>
      ) : (
        <ul className="flex flex-col gap-3 pt-4" aria-label="Expert skills">
          {visible.map((entry) => (
            <li key={entry.name}>
              <ExpertSkillListItem
                entry={entry}
                accentClassName={accentClassName}
                isSaving={isSaving}
                onRemove={removeSkill}
              />
            </li>
          ))}
        </ul>
      )}
      <AddSkillDialog
        open={isAddOpen}
        source={source}
        onSourceChange={setSource}
        skills={available}
        isLoading={isLibraryLoading}
        marketQuery={marketQuery}
        onMarketQueryChange={setMarketQuery}
        marketplaceSkills={marketplaceSkills}
        isMarketplaceLoading={isMarketplaceLoading}
        isSaving={isSaving}
        onAdd={addSkill}
        onAddMarketplace={addMarketplaceSkill}
        onClose={closeAdd}
      />
    </section>
  );
}
