"use client";

import type { DreamOperationsSnapshot } from "@/app/api/__generated__/models/dreamOperationsSnapshot";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/molecules/Accordion/Accordion";
import { Text } from "@/components/atoms/Text/Text";
import {
  FilePlusIcon,
  LightbulbIcon,
  TrashIcon,
  WarningIcon,
} from "@phosphor-icons/react";
import { DreamWriteRow } from "./components/DreamWriteRow";
import { DreamDemotionRow } from "./components/DreamDemotionRow";
import { DreamEntityInvalidationRow } from "./components/DreamEntityInvalidationRow";
import { sectionCounts } from "./helpers";

interface Props {
  operations: DreamOperationsSnapshot | null | undefined;
}

export function DreamOperationsView({ operations }: Props) {
  if (!operations) {
    return (
      <div
        className="rounded-md border border-dashed bg-white p-4 text-center text-sm text-gray-500"
        data-testid="dream-operations-empty"
      >
        No per-edge operations were returned for this dream pass.
      </div>
    );
  }

  const counts = sectionCounts(operations);
  const writes = operations.writes ?? [];
  const proposals = operations.proposals ?? [];
  const demotions = operations.demotions ?? [];
  const entityInvalidations = operations.entity_invalidations ?? [];

  return (
    <div
      className="rounded-md border bg-white p-3"
      data-testid="dream-operations-view"
    >
      <Text variant="small-medium" className="mb-2 uppercase text-gray-500">
        Operations
      </Text>
      <Accordion
        type="multiple"
        defaultValue={["writes", "proposals", "demotions", "entities"]}
      >
        <AccordionItem value="writes">
          <AccordionTrigger>
            <SectionLabel
              icon={<FilePlusIcon size={14} weight="bold" />}
              label="Writes"
              count={counts.writes}
            />
          </AccordionTrigger>
          <AccordionContent>
            {writes.length === 0 ? (
              <EmptyRow text="No writes recorded." />
            ) : (
              <ul className="space-y-1.5">
                {writes.map((w, i) => (
                  <DreamWriteRow
                    key={(w.edge_uuid ?? "w") + i.toString()}
                    item={w}
                  />
                ))}
              </ul>
            )}
          </AccordionContent>
        </AccordionItem>

        <AccordionItem value="proposals">
          <AccordionTrigger>
            <SectionLabel
              icon={<LightbulbIcon size={14} weight="bold" />}
              label="Proposals"
              count={counts.proposals}
            />
          </AccordionTrigger>
          <AccordionContent>
            {proposals.length === 0 ? (
              <EmptyRow text="No proposals recorded." />
            ) : (
              <ul className="space-y-1.5">
                {proposals.map((p, i) => (
                  <DreamWriteRow
                    key={(p.edge_uuid ?? "p") + i.toString()}
                    item={p}
                  />
                ))}
              </ul>
            )}
          </AccordionContent>
        </AccordionItem>

        <AccordionItem value="demotions">
          <AccordionTrigger>
            <SectionLabel
              icon={<TrashIcon size={14} weight="bold" />}
              label="Demotions"
              count={counts.demotions}
            />
          </AccordionTrigger>
          <AccordionContent>
            {demotions.length === 0 ? (
              <EmptyRow text="No demotions recorded." />
            ) : (
              <ul className="space-y-1.5">
                {demotions.map((d) => (
                  <DreamDemotionRow key={d.edge_uuid} item={d} />
                ))}
              </ul>
            )}
          </AccordionContent>
        </AccordionItem>

        <AccordionItem value="entities">
          <AccordionTrigger>
            <SectionLabel
              icon={<WarningIcon size={14} weight="bold" />}
              label="Entity invalidations"
              count={counts.entities}
            />
          </AccordionTrigger>
          <AccordionContent>
            {entityInvalidations.length === 0 ? (
              <EmptyRow text="No entity invalidations recorded." />
            ) : (
              <ul className="space-y-1.5">
                {entityInvalidations.map((e) => (
                  <DreamEntityInvalidationRow key={e.entity_uuid} item={e} />
                ))}
              </ul>
            )}
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </div>
  );
}

interface SectionLabelProps {
  icon: React.ReactNode;
  label: string;
  count: number;
}

function SectionLabel({ icon, label, count }: SectionLabelProps) {
  return (
    <span className="inline-flex items-center gap-2 text-sm font-medium text-gray-800">
      {icon}
      {label}
      <span className="rounded bg-gray-100 px-1.5 py-0.5 text-xs font-normal text-gray-600">
        {count}
      </span>
    </span>
  );
}

function EmptyRow({ text }: { text: string }) {
  return <div className="px-1 py-2 text-xs text-gray-500">{text}</div>;
}
