"use client";

import { CopilotSkillInfo } from "@/app/api/__generated__/models/copilotSkillInfo";
import { Text } from "@/components/atoms/Text/Text";
import { SearchInput } from "@/components/molecules/SearchInput/SearchInput";
import { useState } from "react";
import { ExpertSkillListItem } from "../../[expertId]/components/ExpertSkillListItem";
import { AUTOPILOT_PILL_CLASS } from "../../helpers";

interface Props {
  skills: CopilotSkillInfo[];
}

export function AutopilotSkillsSection({ skills }: Props) {
  const [query, setQuery] = useState("");
  const needle = query.trim().toLowerCase();
  const visible = needle
    ? skills.filter(
        (skill) =>
          skill.name.toLowerCase().includes(needle) ||
          skill.description.toLowerCase().includes(needle),
      )
    : skills;

  return (
    <section>
      <div className="mb-2.5 flex flex-wrap items-center justify-between gap-3">
        <Text variant="body-medium" tone="primary">
          Autopilot&apos;s Skills
        </Text>
        <SearchInput
          size="xsmall"
          value={query}
          onChange={setQuery}
          placeholder="Search skills"
          className="w-48"
        />
      </div>
      {skills.length === 0 ? (
        <Text variant="body" tone="muted" className="pt-4">
          No skills yet. Skills in your library that no expert has claimed show
          up here.
        </Text>
      ) : visible.length === 0 ? (
        <Text variant="body" tone="muted" className="pt-4">
          No skills match.
        </Text>
      ) : (
        <ul className="flex flex-col gap-3 pt-4" aria-label="Autopilot skills">
          {visible.map((skill) => (
            <li key={skill.name}>
              <ExpertSkillListItem
                entry={{ name: skill.name, library: skill }}
                accentClassName={AUTOPILOT_PILL_CLASS}
              />
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
