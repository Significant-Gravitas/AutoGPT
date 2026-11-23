import { RelativeTime } from "@/components/atoms/RelativeTime/RelativeTime";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { Fragment } from "react";
import {
  factText,
  factTime,
  shownFacts,
  type Fact,
  type Reference,
} from "../helpers";

// Named, then a count: the marketplace expert card's skills, in a line.
const NAMED_SKILLS = 3;

interface Props {
  reference: Reference;
}

export function ReferenceCard({ reference }: Props) {
  const facts = reference.kind ? shownFacts(reference.meta) : [];
  const hasFacts = reference.kind
    ? facts.length > 0
    : Boolean(reference.summary);
  const hasAvatar = Boolean(reference.avatarURL || reference.avatarColor);
  return (
    <div data-slot="reference-card">
      <div
        data-slot="reference-card-header"
        className="flex items-center gap-2.5"
      >
        {hasAvatar && (
          <ExpertAvatar
            name={reference.name}
            avatarUrl={reference.avatarURL}
            color={reference.avatarColor}
            size={36}
            className="shrink-0 rounded-full"
          />
        )}
        <div data-slot="reference-card-heading" className="min-w-0">
          {reference.kind && (
            <p
              data-slot="reference-card-kind"
              className="text-foreground-tertiary"
            >
              {reference.kind}
            </p>
          )}
          <p
            translate="no"
            data-slot="reference-card-name"
            className="text-body font-semibold text-foreground-strong"
          >
            {reference.name}
          </p>
        </div>
      </div>
      {reference.description && (
        <p
          data-slot="reference-card-description"
          className="mt-1 line-clamp-3 text-foreground-body"
        >
          {reference.description}
        </p>
      )}
      {hasFacts && (
        <p
          data-slot="reference-card-facts"
          className="mt-1.5 text-foreground-tertiary"
        >
          {reference.kind ? <Facts facts={facts} /> : reference.summary}
        </p>
      )}
      {reference.skills.length > 0 && (
        <p
          data-slot="reference-card-skills"
          className="mt-1.5 text-foreground-body"
        >
          <span
            data-slot="reference-card-skills-label"
            className="text-foreground-tertiary"
          >
            Skills:{" "}
          </span>
          {skillsText(reference.skills)}
        </p>
      )}
      <p
        translate="no"
        data-slot="reference-card-id"
        className="mt-1.5 font-mono text-caption-sm leading-inherit text-foreground-subtle"
      >
        {reference.id}
      </p>
    </div>
  );
}

function Facts({ facts }: { facts: Fact[] }) {
  return facts.map((fact, index) => (
    <Fragment key={index}>
      {index > 0 ? " · " : null}
      <FactText fact={fact} />
    </Fragment>
  ));
}

function FactText({ fact }: { fact: Fact }) {
  const at = factTime(fact);
  if (!at) return factText(fact);
  return (
    <>
      {fact.label} <RelativeTime date={at} />
    </>
  );
}

function skillsText(skills: string[]) {
  const named = skills.slice(0, NAMED_SKILLS).join(", ");
  const rest = skills.length - NAMED_SKILLS;
  return rest > 0 ? `${named} +${rest} more` : named;
}
