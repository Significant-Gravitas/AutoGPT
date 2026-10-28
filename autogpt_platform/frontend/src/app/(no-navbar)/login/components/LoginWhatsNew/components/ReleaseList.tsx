import { Release } from "../helpers";
import { ReleaseItem } from "./ReleaseItem";

interface Props {
  releases: Release[];
}

export function ReleaseList({ releases }: Props) {
  return (
    <ul className="flex flex-col gap-1 rounded-2xl bg-surface p-1.5 smooth-shadow-ring-sm">
      {releases.map((release, index) => (
        <ReleaseItem
          key={release.slug}
          release={release}
          isLatest={index === 0}
          isLast={index === releases.length - 1}
        />
      ))}
    </ul>
  );
}
