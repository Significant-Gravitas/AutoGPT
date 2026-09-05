"use client";

import { CopilotSkillInfo } from "@/app/api/__generated__/models/copilotSkillInfo";
import { StoreAgent } from "@/app/api/__generated__/models/storeAgent";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { SearchInput } from "@/components/molecules/SearchInput/SearchInput";
import {
  TabsLine,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";
import {
  BookOpen01Icon,
  PlusSignIcon,
  Store01Icon,
} from "@hugeicons/core-free-icons";
import NextLink from "next/link";
import { useState } from "react";
import { useFitListToDialog } from "./useFitListToDialog";

type Source = "library" | "marketplace";

interface Props {
  open: boolean;
  source: Source;
  onSourceChange: (source: Source) => void;
  skills: CopilotSkillInfo[];
  isLoading: boolean;
  marketQuery: string;
  onMarketQueryChange: (query: string) => void;
  marketplaceSkills: StoreAgent[];
  isMarketplaceLoading: boolean;
  isSaving: boolean;
  onAdd: (name: string) => void;
  onAddMarketplace: (agent: StoreAgent) => void;
  onClose: () => void;
}

export function AddSkillDialog({
  open,
  source,
  onSourceChange,
  skills,
  isLoading,
  marketQuery,
  onMarketQueryChange,
  marketplaceSkills,
  isMarketplaceLoading,
  isSaving,
  onAdd,
  onAddMarketplace,
  onClose,
}: Props) {
  const [query, setQuery] = useState("");
  const { attachList } = useFitListToDialog<HTMLUListElement>();
  const needle = query.trim().toLowerCase();
  const visible = needle
    ? skills.filter(
        (skill) =>
          skill.name.toLowerCase().includes(needle) ||
          skill.description.toLowerCase().includes(needle),
      )
    : skills;

  return (
    <Dialog
      controlled={{
        isOpen: open,
        set: (next) => {
          if (!next) onClose();
        },
      }}
      styling={{ maxWidth: "30rem", maxHeight: "60vh" }}
      title="Add a skill"
    >
      <Dialog.Content>
        <div className="flex flex-col gap-3">
          <Text variant="small" className="text-sm !text-zinc-500">
            Skills are reusable instructions this expert follows for a specific
            job. Pick one from your library or install one from the marketplace.
          </Text>
          <TabsLine
            value={source}
            onValueChange={(next) => onSourceChange(next as Source)}
          >
            <TabsLineList flush indicatorClassName="bg-zinc-900">
              <TabsLineTrigger
                value="library"
                className="gap-1.5 data-[state=active]:text-zinc-900"
              >
                <Icon icon={BookOpen01Icon} size={14} />
                Library
              </TabsLineTrigger>
              <TabsLineTrigger
                value="marketplace"
                className="gap-1.5 data-[state=active]:text-zinc-900"
              >
                <Icon icon={Store01Icon} size={14} />
                Marketplace
              </TabsLineTrigger>
            </TabsLineList>
          </TabsLine>

          {source === "library" ? (
            <>
              <SearchInput
                size="small"
                value={query}
                onChange={setQuery}
                placeholder="Search your library"
              />
              {isLoading ? (
                <Text variant="small" className="text-zinc-500">
                  Loading your skills…
                </Text>
              ) : visible.length === 0 ? (
                <Text variant="small" className="text-zinc-500">
                  {skills.length === 0
                    ? "Every library skill is already on this expert."
                    : "No skills match."}
                </Text>
              ) : (
                <ul
                  ref={attachList}
                  className="flex flex-col gap-2 overflow-y-auto pr-1"
                  aria-label="Library skills"
                >
                  {visible.map((skill) => (
                    <SkillOption
                      key={skill.name}
                      name={skill.name}
                      description={skill.description}
                      icon={BookOpen01Icon}
                      disabled={isSaving}
                      onAdd={() => onAdd(skill.name)}
                    />
                  ))}
                </ul>
              )}
              <NextLink
                href="/library/skills"
                className="text-sm text-zinc-500 underline underline-offset-2"
              >
                Upload a new skill in your library
              </NextLink>
            </>
          ) : (
            <>
              <SearchInput
                size="small"
                value={marketQuery}
                onChange={onMarketQueryChange}
                placeholder="Search the marketplace"
              />
              {isMarketplaceLoading ? (
                <Text variant="small" className="text-zinc-500">
                  Searching the marketplace…
                </Text>
              ) : marketplaceSkills.length === 0 ? (
                <Text variant="small" className="text-zinc-500">
                  No marketplace skills match.
                </Text>
              ) : (
                <ul
                  ref={attachList}
                  className="flex flex-col gap-2 overflow-y-auto pr-1"
                  aria-label="Marketplace skills"
                >
                  {marketplaceSkills.map((agent) => (
                    <SkillOption
                      key={`${agent.creator}/${agent.slug}`}
                      name={agent.agent_name}
                      description={agent.sub_heading || agent.description}
                      icon={Store01Icon}
                      disabled={isSaving}
                      onAdd={() => onAddMarketplace(agent)}
                    />
                  ))}
                </ul>
              )}
            </>
          )}
          <div className="flex justify-end pt-1">
            <Button variant="secondary" size="small" onClick={onClose}>
              Cancel
            </Button>
          </div>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}

interface OptionProps {
  name: string;
  description: string;
  icon: typeof BookOpen01Icon;
  disabled: boolean;
  onAdd: () => void;
}

function SkillOption({
  name,
  description,
  icon,
  disabled,
  onAdd,
}: OptionProps) {
  return (
    <li className="flex items-center gap-3 rounded-2xl border border-zinc-200 px-4 py-3">
      <Icon icon={icon} size={18} className="shrink-0 text-zinc-500" />
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm font-medium text-zinc-900">
          {name}
        </span>
        <span className="block truncate text-xs text-zinc-500">
          {description}
        </span>
      </span>
      <Button
        variant="secondary"
        size="small"
        disabled={disabled}
        onClick={onAdd}
        leftIcon={<Icon icon={PlusSignIcon} size={14} />}
      >
        Add
      </Button>
    </li>
  );
}
