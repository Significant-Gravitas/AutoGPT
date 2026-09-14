"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { Button } from "@/components/atoms/Button/Button";
import { Switch } from "@/components/atoms/Switch/Switch";
import { Text } from "@/components/atoms/Text/Text";
import { SearchInput } from "@/components/molecules/SearchInput/SearchInput";
import { PlusSignIcon } from "@hugeicons/core-free-icons";
import { AddSkillDialog } from "./AddSkillDialog";
import { ExpertSkillListItem } from "./ExpertSkillListItem";
import { SkillLearningSheet } from "@/components/organisms/SkillLearningSheet/SkillLearningSheet";
import { useExpertLearning } from "./useExpertLearning";
import { useExpertSkills } from "./useExpertSkills";

interface Props {
  expert: Expert;
  accentClassName?: string;
  initialSkill?: string | null;
  initialVersionId?: string | null;
}

export function ExpertSkillsSection({
  expert,
  accentClassName,
  initialSkill = null,
  initialVersionId = null,
}: Props) {
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
  const learning = useExpertLearning(expert, initialSkill, initialVersionId);

  return (
    <section>
      <div className="mb-2.5 flex flex-wrap items-center justify-between gap-3">
        <Text variant="large-medium" tone="primary">
          {expert.name}&apos;s Skills
        </Text>
        <div className="flex items-center gap-2">
          <Button
            variant="secondary"
            size="small"
            leadingIcon={PlusSignIcon}
            onClick={openAdd}
          >
            Add skill
          </Button>
          <SearchInput
            size="small"
            value={query}
            onChange={setQuery}
            placeholder="Search skills"
            className="w-48"
          />
        </div>
      </div>
      {learning.enabled ? (
        <div
          className="mb-3 flex flex-col gap-2 rounded-2xl bg-white p-3.5 smooth-shadow-ring-sm sm:flex-row sm:items-center sm:justify-between"
          data-testid="expert-learning-summary"
        >
          <Text variant="small" tone="secondary">
            {learning.recentChange
              ? `${learning.recentChange.summary} · ${learning.recentChange.origin_label ?? ""} · ${learning.recentChange.state_label}`
              : "No overnight changes yet. Verified procedures from chats are reviewed nightly."}
          </Text>
          <label className="flex items-center gap-2">
            <Text variant="small" tone="primary">
              {learning.isLearningPaused ? "Learning paused" : "Learning on"}
            </Text>
            <Switch
              checked={!learning.isLearningPaused}
              disabled={learning.isTogglingLearning}
              aria-label={`Nightly learning for ${expert.name}`}
              onCheckedChange={(checked) =>
                learning.setLearningPaused(!checked)
              }
            />
          </label>
        </div>
      ) : null}
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
                learning={
                  learning.enabled ? learning.learningLineFor(entry.name) : null
                }
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
      {learning.enabled ? (
        <SkillLearningSheet
          scope={{
            expertId: expert.id,
            name: expert.name,
            avatarUrl: expert.avatar_url,
            color: expert.color,
          }}
          skillName={learning.openSkill?.name ?? null}
          initialVersionId={learning.openSkill?.versionId ?? null}
          onChanged={learning.refreshHistory}
          onClose={learning.closeSkill}
        />
      ) : null}
    </section>
  );
}
