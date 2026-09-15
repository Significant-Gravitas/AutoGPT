"use client";

import { Button } from "@/components/atoms/Button/Button";
import { ImportExpertButton } from "./ImportExpertButton/ImportExpertButton";

interface Props {
  /** ``EXPERT_PORTABILITY``: the import routes 404 while it is off. */
  canImport: boolean;
}

export function TeamHeaderActions({ canImport }: Props) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button as="NextLink" href="/raise" variant="secondary" size="small">
        Raise expert
      </Button>
      {canImport ? <ImportExpertButton /> : null}
      <Button
        as="NextLink"
        href="/marketplace#experts"
        variant="primary"
        size="small"
      >
        Hire expert
      </Button>
    </div>
  );
}
