"use client";

import { Text } from "@/components/atoms/Text/Text";
import { ExpandedExpert } from "./components/ExpandedExpert";
import { MarqueeRow } from "./components/MarqueeRow";
import { chunkIntoRows, PROFESSIONS } from "./helpers";
import { useExpertPreview } from "./useExpertPreview";

const ROW_COUNT = 4;
const ROW_DURATIONS = [46, 58, 52, 64];
const ROWS = chunkIntoRows(PROFESSIONS, ROW_COUNT);

export default function ExpertPreviewPage() {
  const { selected, select, close } = useExpertPreview();

  function renderRow(rowIndex: number) {
    return (
      <MarqueeRow
        key={rowIndex}
        rowIndex={rowIndex}
        professions={ROWS[rowIndex]}
        reverse={rowIndex % 2 === 1}
        durationSeconds={ROW_DURATIONS[rowIndex]}
        paused={selected !== null}
        onSelect={select}
      />
    );
  }

  return (
    <main className="flex min-h-screen flex-col justify-center gap-10 overflow-hidden bg-white py-12">
      <div className="flex flex-col gap-8">{[0, 1].map(renderRow)}</div>
      <div className="px-6 text-center">
        <Text variant="h1">Expert Preview</Text>
      </div>
      <div className="flex flex-col gap-8">{[2, 3].map(renderRow)}</div>
      <ExpandedExpert selected={selected} onClose={close} />
    </main>
  );
}
