"use client";

import { PencilEdit02Icon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { ProfileEntry } from "./ProfileEntry";

interface Props {
  identity: string;
  voicePreferences: string | null;
  boundaries: string | null;
  onEdit: () => void;
}

/** The expert's Soul: who it is, how it speaks and what it won't do. */
export function ExpertSoulSection({
  identity,
  voicePreferences,
  boundaries,
  onEdit,
}: Props) {
  return (
    <section className="space-y-5">
      <div className="flex justify-end">
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
