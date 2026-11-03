"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Switch } from "@/components/atoms/Switch/Switch";
import { Text } from "@/components/atoms/Text/Text";
import { SkillLearningSheet } from "@/components/organisms/SkillLearningSheet/SkillLearningSheet";
import {
  BookOpen01Icon,
  BulbIcon,
  PlusSignIcon,
  SparklesIcon,
} from "@hugeicons/core-free-icons";
import { SectionEmptyState } from "../../components/SectionEmptyState";
import { ToolbarSearch } from "../../components/ToolbarSearch";
import { AddSkillDialog } from "./AddSkillDialog";
import { ExpertSkillListItem } from "./ExpertSkillListItem";
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
        <div className="flex w-full flex-wrap items-center gap-2 sm:w-auto">
          <Button
            variant="secondary"
            size="small"
            textSize="body-md"
            leadingIcon={PlusSignIcon}
            onClick={openAdd}
          >
            Add skill
          </Button>
          <ToolbarSearch
            textSize="body-md"
            id="expert-skills-search"
            label="Search skills"
            value={query}
            onChange={setQuery}
          />
        </div>
      </div>
      {learning.enabled ? (
        <div
          className="mb-3 flex flex-col gap-2 rounded-2xl bg-surface p-3.5 smooth-shadow-ring-sm sm:flex-row sm:items-center sm:justify-between"
          data-testid="expert-learning-summary"
        >
          <Text variant="small" tone="secondary">
            {learning.recentChange
              ? `${learning.recentChange.summary} · ${learning.recentChange.origin_label ?? ""} · ${learning.recentChange.state_label}`
              : "No overnight changes yet. Verified procedures from chats are reviewed nightly."}
          </Text>
          <label className="flex shrink-0 items-center gap-2 whitespace-nowrap">
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
        <SectionEmptyState
          tiles={[
            // oxlint-disable-next-line design-system/no-raw-colors -- sky has no role; info blue would change the hue (deferred)
            <span key="book" className="inline-flex text-sky-500">
              <Icon icon={BookOpen01Icon} size={22} />
            </span>,
            <span key="bulb" className="inline-flex text-warning">
              <Icon icon={BulbIcon} size={22} />
            </span>,
            <span key="sparkles" className="inline-flex text-brand">
              <Icon icon={SparklesIcon} size={22} />
            </span>,
          ]}
          title="No skills yet"
          description={`Add skills from your library so ${expert.name} knows how you like things done.`}
        >
          <Button
            variant="primary"
            size="small"
            textSize="body-md"
            leadingIcon={PlusSignIcon}
            onClick={openAdd}
          >
            Add skill
          </Button>
        </SectionEmptyState>
      ) : visible.length === 0 ? (
        <Text variant="body-md" tone="muted" className="mt-4">
          No skills match.
        </Text>
      ) : (
        <ul
          className="grid grid-cols-1 gap-3 pt-4 md:grid-cols-2"
          aria-label="Expert skills"
        >
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
