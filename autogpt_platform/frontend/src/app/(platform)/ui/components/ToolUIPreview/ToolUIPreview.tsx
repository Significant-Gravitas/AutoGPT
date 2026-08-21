"use client";

import { Text } from "@/components/atoms/Text/Text";
import { AttentionRow } from "../../../home/components/NeedsYou/components/AttentionRow";
import { RowIcon } from "../../../copilot/components/ToolChain/RowIcon";
import { ToolResult } from "../../../copilot/components/ToolChain/ToolResult";
import { SAMPLE_QUESTION_ITEM, SAMPLE_ROWS } from "../../samples";

export function ToolUIPreview() {
  return (
    <main className="mx-auto flex w-full max-w-3xl flex-col gap-10 px-6 py-12">
      <header className="flex flex-col gap-1">
        <Text variant="h3">Expert tool UI</Text>
        <Text variant="body" className="text-zinc-500">
          Chain rows and result cards for hire_expert, raise_expert,
          confirm_expert_change and handoff_to_expert, plus the Home question
          item — rendered from fixtures, no backend needed.
        </Text>
      </header>

      {SAMPLE_ROWS.map((sample) => (
        <section key={sample.row.key} className="flex flex-col gap-2">
          <Text variant="large-medium">{sample.title}</Text>
          <div className="flex items-center gap-2 rounded-xl bg-zinc-50 px-3 py-2">
            <RowIcon row={sample.row} />
            <Text variant="small" className="text-zinc-700">
              {sample.label}
            </Text>
          </div>
          <ToolResult row={sample.row} />
        </section>
      ))}

      <section className="flex flex-col gap-2">
        <Text variant="large-medium">Home · Needs You question</Text>
        <div className="rounded-xl bg-white ring-1 ring-zinc-200/70">
          <AttentionRow
            item={SAMPLE_QUESTION_ITEM}
            isProcessing={false}
            onDecision={() => undefined}
          />
        </div>
      </section>
    </main>
  );
}
