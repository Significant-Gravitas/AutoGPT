import { Fragment } from "react";
import type { Reference } from "../helpers";
import { ReferenceLink } from "./ReferenceLink";

interface Props {
  refs: Reference[];
  // How many ids the argument held; the server names the first few.
  total: number;
}

export function ReferenceValue({ refs, total }: Props) {
  const more = total - refs.length;
  return (
    <span>
      {refs.map((ref, i) => (
        <Fragment key={`${ref.id}-${i}`}>
          {i > 0 && ", "}
          <ReferenceName reference={ref} />
        </Fragment>
      ))}
      {more > 0 && <span className="text-zinc-500"> +{more} more</span>}
    </span>
  );
}

function ReferenceName({ reference }: { reference: Reference }) {
  if (!reference.name)
    return (
      <span translate="no" className="font-mono text-[0.8125rem]">
        {reference.id}
      </span>
    );
  return <ReferenceLink reference={reference}>{reference.name}</ReferenceLink>;
}
