"use client";

import { Text } from "@/components/atoms/Text/Text";

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

interface ProfileEntryProps {
  label: string;
  value: string | null;
}

function ProfileEntry({ label, value }: ProfileEntryProps) {
  return (
    <div>
      <Text variant="body-medium" as="dt" tone="primary">
        {label}
      </Text>
      <Text
        variant="body"
        as="dd"
        tone="secondary"
        className="mt-1 whitespace-pre-line"
      >
        {value || "Not set yet."}
      </Text>
    </div>
  );
}
