import Link from "next/link";
import { Fragment } from "react";
import type { Reference } from "../helpers";

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
  if (!reference.href)
    return (
      <span translate="no" title={reference.id}>
        {reference.name}
      </span>
    );
  return (
    <Link
      href={reference.href}
      translate="no"
      title={reference.id}
      className="underline decoration-zinc-300 underline-offset-2 hover:text-zinc-600 hover:decoration-zinc-500 focus-visible:rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-300"
    >
      {reference.name}
    </Link>
  );
}
