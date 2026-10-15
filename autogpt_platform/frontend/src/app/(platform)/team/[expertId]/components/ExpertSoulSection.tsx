"use client";

import { PencilEdit02Icon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { ProfileEntry } from "./ProfileEntry";

interface Props {
  expertName: string;
  identity: string;
  voicePreferences: string | null;
  boundaries: string | null;
  onEdit: () => void;
}

/** The expert's Soul: who it is, how it speaks and what it won't do. */
export function ExpertSoulSection({
  expertName,
  identity,
  voicePreferences,
  boundaries,
  onEdit,
}: Props) {
  return (
    <section className="space-y-5">
      <div className="flex items-center justify-between gap-4">
        <Text variant="body" tone="muted">
          How {expertName} thinks, speaks and where it draws the line.
        </Text>
        <Button
          variant="secondary"
          size="small"
          leadingIcon={PencilEdit02Icon}
          onClick={onEdit}
        >
          Edit Soul
        </Button>
      </div>
      <dl className="space-y-4">
        <ProfileEntry label="Identity" value={identity} />
        <ProfileEntry label="Voice" value={voicePreferences} />
        <ProfileEntry label="Boundaries" value={boundaries} />
      </dl>
    </section>
  );
}
