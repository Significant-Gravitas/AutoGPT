"use client";

import { useState } from "react";
import { isChainableToolPart } from "../components/ChatMessagesContainer/helpers";
import { MessagePartRenderer } from "../components/ChatMessagesContainer/components/MessagePartRenderer";
import { CopilotChatActionsProvider } from "../components/CopilotChatActionsProvider/CopilotChatActionsProvider";
import { ChainRowView } from "../components/ToolChain/ChainRowView";
import {
  buildChainSegments,
  toChainRow,
} from "../components/ToolChain/helpers";
import { ToolChain } from "../components/ToolChain/ToolChain";
import {
  CATALOG_SECTIONS,
  CHAIN_DEMOS,
  INTERACTIVE_SAMPLES,
  INTERRUPT_DEMO,
  STATE_SAMPLES,
  THINKING_SAMPLES,
  type SampleTool,
  toPart,
} from "./sampleTools";

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="mb-10">
      <h2 className="mb-4 border-b border-zinc-200 pb-2 text-sm font-semibold uppercase tracking-wide text-zinc-500">
        {title}
      </h2>
      {children}
    </section>
  );
}

function RawData({ sample }: { sample: SampleTool }) {
  return (
    <pre className="mt-1 overflow-x-auto rounded-xl bg-zinc-900 p-3 font-mono text-[11px] leading-4 text-zinc-100">
      {JSON.stringify(sample, null, 2)}
    </pre>
  );
}

function RowDemo({
  sample,
  index,
  showRaw,
}: {
  sample: SampleTool;
  index: number;
  showRaw: boolean;
}) {
  const row = toChainRow(toPart(sample, index), index);
  if (!row) return null;
  return (
    <div className="py-1">
      <ChainRowView row={row} isLast />
      {showRaw && <RawData sample={sample} />}
    </div>
  );
}

function MixedSegments({ samples }: { samples: SampleTool[] }) {
  const parts = samples.map(toPart);
  const segments = buildChainSegments(parts, isChainableToolPart);
  return (
    <div className="flex flex-col gap-2">
      {segments.map((segment) =>
        segment.kind === "chain" ? (
          <ToolChain
            key={`chain-${segment.index}`}
            parts={segment.parts}
            isStreaming={false}
          />
        ) : (
          <MessagePartRenderer
            key={`part-${segment.index}`}
            part={segment.part}
            messageID="interrupt-demo"
            partIndex={segment.index}
          />
        ),
      )}
    </div>
  );
}

export function TestUiPage() {
  const [showRaw, setShowRaw] = useState(false);
  const [sentMessage, setSentMessage] = useState<string | null>(null);

  return (
    <CopilotChatActionsProvider onSend={(message) => setSentMessage(message)}>
      <div className="min-h-screen bg-[#fafafa]">
        <button
          type="button"
          onClick={() => setShowRaw(!showRaw)}
          className="fixed right-6 top-6 z-10 rounded-full bg-zinc-900 px-4 py-2 text-xs font-medium text-white shadow-lg transition-colors hover:bg-zinc-700"
        >
          {showRaw ? "Hide raw data" : "Show raw data"}
        </button>

        <div className="mx-auto max-w-3xl px-6 py-10">
          <h1 className="mb-1 text-lg font-semibold text-zinc-900">
            Tool UI — full catalog
          </h1>
          <p className="mb-8 text-sm text-zinc-500">
            Every tool row with random data. Click any row for its input/output;
            click chain headings to expand.
          </p>

          <Section title="Chains">
            <div className="flex flex-col gap-6">
              {CHAIN_DEMOS.map((demo) => (
                <div key={demo.title}>
                  <p className="mb-1 text-xs text-zinc-400">{demo.title}</p>
                  <ToolChain
                    parts={demo.tools.map(toPart)}
                    isStreaming={demo.streaming}
                  />
                  {showRaw &&
                    demo.tools.map((sample, i) => (
                      <RawData key={i} sample={sample} />
                    ))}
                </div>
              ))}
            </div>
          </Section>

          <Section title="Interactive — needs user action (renders outside the chain)">
            {sentMessage && (
              <pre className="mb-3 whitespace-pre-wrap rounded-xl bg-zinc-900 p-3 font-mono text-[11px] leading-4 text-green-300">
                onSend → {sentMessage}
              </pre>
            )}
            <div className="flex flex-col gap-6">
              {INTERACTIVE_SAMPLES.map((entry, i) => (
                <div key={entry.label}>
                  <p className="mb-1 text-xs text-zinc-400">{entry.label}</p>
                  <MessagePartRenderer
                    part={toPart(entry.sample, i)}
                    messageID={`interactive-${i}`}
                    partIndex={i}
                  />
                  {showRaw && <RawData sample={entry.sample} />}
                </div>
              ))}
            </div>
          </Section>

          <Section title="Interactive mid-chain — chain splits around the card">
            <MixedSegments samples={INTERRUPT_DEMO} />
          </Section>

          <Section title="States (streaming input → running → done → error)">
            {STATE_SAMPLES.map((sample, i) => (
              <RowDemo key={i} sample={sample} index={i} showRaw={showRaw} />
            ))}
          </Section>

          <Section title="Thinking (live stream + collapsed 'Thought')">
            {THINKING_SAMPLES.map((sample, i) => (
              <RowDemo key={i} sample={sample} index={i} showRaw={showRaw} />
            ))}
          </Section>

          {CATALOG_SECTIONS.map((section) => (
            <Section key={section.title} title={section.title}>
              {section.tools.map((sample, i) => (
                <RowDemo
                  key={`${sample.tool}-${i}`}
                  sample={sample}
                  index={i}
                  showRaw={showRaw}
                />
              ))}
            </Section>
          ))}
        </div>
      </div>
    </CopilotChatActionsProvider>
  );
}
