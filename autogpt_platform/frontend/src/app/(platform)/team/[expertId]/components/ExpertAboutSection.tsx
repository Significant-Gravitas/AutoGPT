"use client";

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
    <section className="space-y-6">
      {bio ? (
        <p className="whitespace-pre-line text-base leading-relaxed text-zinc-600">
          {bio}
        </p>
      ) : null}

      <dl className="space-y-5">
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
      <dt className="text-sm font-medium text-zinc-900">{label}</dt>
      <dd className="mt-1 whitespace-pre-line text-base leading-relaxed text-zinc-600">
        {value || "Not set yet."}
      </dd>
    </div>
  );
}
