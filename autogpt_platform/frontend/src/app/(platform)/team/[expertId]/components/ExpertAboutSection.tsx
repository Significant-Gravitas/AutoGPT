"use client";

import { ProfileEntry } from "./ProfileEntry";

interface Props {
  bio: string | null;
  identity: string;
  voicePreferences: string | null;
  boundaries: string | null;
}

export function ExpertAboutSection({
  bio,
  identity,
  voicePreferences,
  boundaries,
}: Props) {
  return (
    <section className="space-y-5">
      <dl className="space-y-4">
        <ProfileEntry label="Bio" value={bio} />
        <ProfileEntry label="Identity" value={identity} />
        <ProfileEntry label="Voice" value={voicePreferences} />
        <ProfileEntry label="Boundaries" value={boundaries} />
      </dl>
    </section>
  );
}
