"use client";

import { ProfileEntry } from "./ProfileEntry";

interface Props {
  bio: string | null;
}

/** The public-facing summary. The Soul (identity, voice, boundaries) has
 *  its own tab, see ExpertSoulSection. */
export function ExpertAboutSection({ bio }: Props) {
  return (
    <section className="space-y-5">
      <dl className="space-y-4">
        <ProfileEntry label="Bio" value={bio} />
      </dl>
    </section>
  );
}
