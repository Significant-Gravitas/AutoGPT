import type { Expert } from "@/app/api/__generated__/models/expert";
import type { TaskCreateRequest } from "@/app/api/__generated__/models/taskCreateRequest";
import {
  AUTOPILOT_AVATAR_URL,
  AUTOPILOT_NAME,
} from "@/components/molecules/AutopilotAvatar/helpers";

export const OTTO_VALUE = "otto";
const TITLE_LIMIT = 80;

export interface OwnerChoice {
  /** `otto`, or the expert's id. */
  value: string;
  label: string;
  avatarUrl: string | null;
}

export function ownerChoices(experts: Expert[]): OwnerChoice[] {
  return [
    {
      value: OTTO_VALUE,
      label: AUTOPILOT_NAME,
      avatarUrl: AUTOPILOT_AVATAR_URL,
    },
    ...experts.map((expert) => ({
      value: expert.id,
      label: expert.name,
      avatarUrl: expert.avatar_url,
    })),
  ];
}

export function resolveChoice(
  choices: OwnerChoice[],
  value: string,
): OwnerChoice {
  return choices.find((choice) => choice.value === value) ?? choices[0];
}

/** The first line is the title, shortened to fit; the whole text becomes
 *  the description when it says more than the title. */
export function buildCreateRequest(
  text: string,
  owner: OwnerChoice,
): TaskCreateRequest {
  const full = text.trim();
  const title = shorten(full.split("\n")[0].trim());
  return {
    title,
    description: title === full ? undefined : full,
    owner_expert_id: owner.value === OTTO_VALUE ? undefined : owner.value,
  };
}

function shorten(line: string) {
  if (line.length <= TITLE_LIMIT) return line;
  const cut = line.slice(0, TITLE_LIMIT);
  const lastSpace = cut.lastIndexOf(" ");
  const kept = lastSpace > TITLE_LIMIT / 2 ? cut.slice(0, lastSpace) : cut;
  return `${kept.trimEnd()}…`;
}
