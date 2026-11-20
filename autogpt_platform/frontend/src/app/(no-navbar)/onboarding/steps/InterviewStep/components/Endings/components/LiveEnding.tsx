"use client";

import type { InterviewDetails } from "@/app/api/__generated__/models/interviewDetails";
import { Button } from "@/components/atoms/Button/Button";
import { profileRowsOf } from "../helpers";
import { CoverageRing } from "./CoverageRing";
import { ProfileCardEnding } from "./ProfileCardEnding";

interface Props {
  arm: string | null;
  details: InterviewDetails | null;
  canFinish: boolean;
  isLeaving: boolean;
  onFinish: () => void;
}

// What an ending shows while the typed interview runs: the profile card
// filling in, or the topics ticking off with Done there from the start.
export function LiveEnding({
  arm,
  details,
  canFinish,
  isLeaving,
  onFinish,
}: Props) {
  if (arm === "02-profile-card") {
    const rows = profileRowsOf(details);
    return rows.length > 0 ? <ProfileCardEnding rows={rows} isLive /> : null;
  }
  if (arm === "06-coverage-ring") {
    return (
      <div className="flex w-full flex-col items-center gap-3">
        <CoverageRing covered={details?.topics ?? []} />
        <Button
          variant="secondary"
          size="small"
          loading={isLeaving}
          disabled={!canFinish || isLeaving}
          onClick={onFinish}
        >
          Done, build my team
        </Button>
      </div>
    );
  }
  return null;
}
