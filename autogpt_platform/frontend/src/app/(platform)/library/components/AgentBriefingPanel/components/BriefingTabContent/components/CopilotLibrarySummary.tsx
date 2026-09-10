"use client";

import { useListCopilotSkills } from "@/app/api/__generated__/endpoints/skills/skills";
import { useListCopilotFollowupSchedules } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { Text } from "@/components/atoms/Text/Text";
import Link from "next/link";

export function CopilotLibrarySummary() {
  // Discoverability is already gated by AGENT_BRIEFING at the parent
  // panel — this pill renders only inside AgentBriefingPanel, which is
  // itself flag-gated.  No second flag here because the count-based
  // hide below already keeps the pill quiet for users who don't use
  // the feature.
  const { data: skillsRes } = useListCopilotSkills({
    query: { staleTime: 30_000 },
  });
  const { data: followupsRes } = useListCopilotFollowupSchedules({
    query: { staleTime: 30_000 },
  });

  const skillsCount =
    skillsRes && skillsRes.status === 200 ? skillsRes.data.length : 0;
  // Count only copilot follow-ups here — graph schedules (recurring
  // agent runs) are already surfaced by the briefing's own "Scheduled"
  // tab above, so folding them into this pill would double-count and
  // confuse the "Autopilot library" framing.  The pill's link still
  // goes to the unified `/library/followups` page, where both kinds
  // are listed together.
  const followupCount =
    followupsRes && followupsRes.status === 200 ? followupsRes.data.length : 0;

  // Suppress the pill entirely when the user has no autopilot library
  // content yet — surfacing "0 skills · 0 follow-ups" is noise, not a
  // discovery affordance.  The pill reappears the moment either count
  // turns positive (e.g. after a store_skill / schedule_followup tool
  // call).
  if (skillsCount === 0 && followupCount === 0) return null;

  // Per-link hide: surface only the counts that are non-zero.  We
  // already returned ``null`` above when both are zero, so at least
  // one branch always renders.
  const showSkills = skillsCount > 0;
  const showFollowups = followupCount > 0;

  return (
    <div
      className="mt-5 flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-zinc-100 pt-3"
      data-testid="copilot-library-summary"
    >
      <Text variant="small" className="!text-zinc-500">
        Autopilot library
      </Text>
      {showSkills ? (
        <Link
          href="/library/skills"
          className="text-sm text-violet-700 hover:underline"
          data-testid="copilot-library-skills-link"
        >
          {skillsCount} skill{skillsCount === 1 ? "" : "s"}
        </Link>
      ) : null}
      {showSkills && showFollowups ? (
        <span className="text-zinc-300">•</span>
      ) : null}
      {showFollowups ? (
        <Link
          href="/library/followups"
          className="text-sm text-yellow-700 hover:underline"
          data-testid="copilot-library-followups-link"
        >
          {followupCount} follow-up{followupCount === 1 ? "" : "s"}
        </Link>
      ) : null}
    </div>
  );
}
