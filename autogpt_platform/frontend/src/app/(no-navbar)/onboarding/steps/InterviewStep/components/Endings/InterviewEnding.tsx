"use client";

import type { InterviewDetails } from "@/app/api/__generated__/models/interviewDetails";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import type { ReactNode } from "react";
import { CoverageRing } from "./components/CoverageRing";
import { FirstTaskEnding } from "./components/FirstTaskEnding";
import { HighlightsEnding } from "./components/HighlightsEnding";
import { ProfileCardEnding } from "./components/ProfileCardEnding";
import { SummaryChipsEnding } from "./components/SummaryChipsEnding";
import { TeamReveal } from "./components/TeamReveal";
import {
  chipsOf,
  type EndingArm,
  ENDING_NAMES,
  profileRowsOf,
} from "./helpers";

export interface EndingData {
  firstTask: string | null;
  details?: InterviewDetails | null;
}

interface Props {
  arm: EndingArm | null;
  data: EndingData;
  isLeaving: boolean;
  isTesterPick: boolean;
  onConfirm: (corrections: string | null) => void;
  // Today's ending, for anyone not enrolled, an arm with no screen here, and
  // an arm whose data did not come.
  fallback: ReactNode;
}

// How the interview ends on screen, by the person's arm of the ending
// experiment; shared by the call and the typed interview.
export function InterviewEnding({
  arm,
  data,
  isLeaving,
  isTesterPick,
  onConfirm,
  fallback,
}: Props) {
  const screen = armScreen({ arm, data, isLeaving, onConfirm, fallback });
  return (
    <div className="flex w-full flex-col items-center gap-3">
      {screen ?? fallback}
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
  fallback,
}: Omit<Props, "isTesterPick">) {
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
    case "03-summary-chips": {
      const chips = chipsOf(details);
      if (chips.length === 0) return null;
      return (
        <SummaryChipsEnding
          chips={chips}
          isLeaving={isLeaving}
          onConfirm={onConfirm}
        />
      );
    }
    case "04-team-reveal":
      return (
        <TeamReveal
          isLeaving={isLeaving}
          onMeet={() => onConfirm(null)}
          fallback={fallback}
        />
      );
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
    case "08-transcript-highlights": {
      const words = details?.words ?? [];
      const facts = details?.facts ?? [];
      if (words.length === 0 || facts.length === 0) return null;
      return (
        <HighlightsEnding
          words={words}
          facts={facts}
          isLeaving={isLeaving}
          onConfirm={onConfirm}
        />
      );
    }
    case "09-first-task":
      if (!data.firstTask) return null;
      return (
        <FirstTaskEnding
          firstTask={data.firstTask}
          owner={null}
          isLeaving={isLeaving}
          onConfirm={onConfirm}
        />
      );
    default:
      return null;
  }
}
