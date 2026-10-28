import { Release } from "../helpers";
import { ReleaseItem } from "./ReleaseItem";
import { RELEASE_LIST_CLASS } from "./releaseRowLayout";

interface Props {
  releases: Release[];
}

export function ReleaseList({ releases }: Props) {
  return (
    <ul className={RELEASE_LIST_CLASS}>
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
