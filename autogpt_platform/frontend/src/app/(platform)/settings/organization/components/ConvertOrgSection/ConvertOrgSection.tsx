"use client";

import type { OrgResponse } from "@/app/api/__generated__/models/orgResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

import { CONVERT_CONSEQUENCES } from "./helpers";
import { useConvertOrgSection } from "./useConvertOrgSection";

interface Props {
  org: OrgResponse;
}

export function ConvertOrgSection({ org }: Props) {
  const {
    isConfirmOpen,
    setIsConfirmOpen,
    isConverting,
    handleConvertConfirmed,
  } = useConvertOrgSection({ org });

  return (
    <section
      className="flex flex-wrap items-center justify-between gap-3 rounded-large border border-border-base p-4"
      data-testid="org-convert-section"
    >
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <Text variant="body-medium" as="h2">
          Turn this into a team organization
        </Text>
        <Text variant="small" tone="muted">
          Keep what you&apos;ve built here and invite people to work on it with
          you.
        </Text>
      </div>
      <Button
        variant="secondary"
        size="small"
        onClick={() => setIsConfirmOpen(true)}
      >
        Convert to a team organization
      </Button>

      <Dialog
        title={`Convert ${org.name} into a team organization?`}
        styling={{ maxWidth: "30rem" }}
        controlled={{
          isOpen: isConfirmOpen,
          set: (open) => {
            if (!isConverting) setIsConfirmOpen(open);
          },
        }}
      >
        <Dialog.Content>
          <div className="flex flex-col gap-3">
            <ul className="flex list-disc flex-col gap-2 pl-5">
              {CONVERT_CONSEQUENCES.map((consequence) => (
                <li key={consequence}>
                  <Text variant="body" as="span">
                    {consequence}
                  </Text>
                </li>
              ))}
            </ul>
            <Text variant="small" tone="muted">
              You can&apos;t turn it back into a personal organization.
            </Text>
          </div>
          <Dialog.Footer>
            <Button
              variant="secondary"
              size="small"
              onClick={() => setIsConfirmOpen(false)}
              disabled={isConverting}
            >
              Cancel
            </Button>
            <Button
              size="small"
              loading={isConverting}
              onClick={() => void handleConvertConfirmed()}
            >
              Convert organization
            </Button>
          </Dialog.Footer>
        </Dialog.Content>
      </Dialog>
    </section>
  );
}
