"use client";

import { ProfileEntry } from "./ProfileEntry";

interface Props {
  identity: string;
  voicePreferences: string | null;
  boundaries: string | null;
}

/** The expert's Soul: who it is, how it speaks and what it won't do. The
 *  header's "Edit Soul" button edits it. */
export function ExpertSoulSection({
  identity,
  voicePreferences,
  boundaries,
}: Props) {
  return (
    <dl className="space-y-4">
      <ProfileEntry label="Identity" value={identity} />
      <ProfileEntry label="Voice" value={voicePreferences} />
      <ProfileEntry label="Boundaries" value={boundaries} />
    </dl>
  );
}
