"use client";

import type { InterviewDetails } from "@/app/api/__generated__/models/interviewDetails";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import type { ReactNode } from "react";
import { CoverageRing } from "./components/CoverageRing";
import { ProfileCardEnding } from "./components/ProfileCardEnding";
import { SettledEnding } from "./components/SettledEnding";
import {
  type EndingArm,
  ENDING_NAMES,
  profileRowsOf,
  settledViewOf,
} from "./helpers";

export interface EndingData {
  details?: InterviewDetails | null;
}

interface Props {
  arm: EndingArm | null;
  data: EndingData;
  isLeaving: boolean;
  isTesterPick: boolean;
  // The step opened on an interview already over: the profile card shows
  // read-only, and asks nothing again.
  isSettled?: boolean;
  onConfirm: (corrections: string | null) => void;
  // The edits could not be saved: the way on without them.
  onLeaveWithoutChanges?: () => void;
  // Today's ending, for anyone not enrolled, an arm with no screen here, and
  // an arm whose data did not come.
  fallback: ReactNode;
  // The way to do the interview over, under the ending.
  startOver?: ReactNode;
}

// How the interview ends on screen, by the person's arm of the ending
// experiment; shared by the call and the typed interview.
export function InterviewEnding({
  arm,
  data,
  isLeaving,
  isTesterPick,
  isSettled = false,
  onConfirm,
  onLeaveWithoutChanges,
  fallback,
  startOver,
}: Props) {
  const isSettledCard = isSettled && arm === "02-profile-card";
  const settled = isSettledCard
    ? settledViewOf(arm, data.details ?? null)
    : null;
  const screen = isSettledCard
    ? settled && (
        <SettledEnding
          view={settled}
          isLeaving={isLeaving}
          onContinue={() => onConfirm(null)}
        />
      )
    : armScreen({ arm, data, isLeaving, onConfirm });
  return (
    <div className="flex w-full flex-col items-center gap-3">
      {screen ?? fallback}
      {onLeaveWithoutChanges && (
        <Button
          variant="secondary"
          size="small"
          disabled={isLeaving}
          onClick={onLeaveWithoutChanges}
        >
          Continue without my changes
        </Button>
      )}
      {startOver}
      {arm && isTesterPick && (
        <Text variant="small" tone="muted" align="center">
          Ending: {ENDING_NAMES[arm]} (tester pick)
        </Text>
      )}
    </div>
  );
}

function armScreen({
  arm,
  data,
  isLeaving,
  onConfirm,
}: Omit<
  Props,
  | "isTesterPick"
  | "isSettled"
  | "startOver"
  | "fallback"
  | "onLeaveWithoutChanges"
>) {
  const details = data.details ?? null;
  switch (arm) {
    case "02-profile-card": {
      const rows = profileRowsOf(details);
      if (rows.length === 0) return null;
      return (
        <ProfileCardEnding
          key={rows.map((row) => row.value).join("|")}
          rows={rows}
          isLeaving={isLeaving}
          onConfirm={onConfirm}
        />
      );
    }
    case "06-coverage-ring":
      return (
        <div className="flex w-full max-w-md flex-col items-center gap-4">
          <CoverageRing covered={details?.topics ?? []} />
          <Button
            variant="primary"
            size="large"
            loading={isLeaving}
            disabled={isLeaving}
            onClick={() => onConfirm(null)}
          >
            Done, build my team
          </Button>
        </div>
      );
    default:
      return null;
  }
}
