"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
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
        <Text variant="large-medium">{expert.name}&apos;s Skills</Text>
        <div className="flex items-center gap-2">
          <Button
            variant="secondary"
            size="small"
            leftIcon={<Icon icon={PlusSignIcon} size={16} />}
            onClick={openAdd}
          >
            Add skill
          </Button>
          <SearchInput
            size="small"
            value={query}
            onChange={setQuery}
            placeholder="Search skills"
            className="w-56"
          />
        </div>
      </div>
      {!hasAny ? (
        <p className="pt-4 text-sm text-zinc-500">
          No skills yet. Add skills from your library so {expert.name} knows how
          you like things done.
        </p>
      ) : visible.length === 0 ? (
        <p className="pt-4 text-sm text-zinc-500">No skills match.</p>
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
