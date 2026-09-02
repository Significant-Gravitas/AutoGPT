"use client";

import { useGetV1ListProviders } from "@/app/api/__generated__/endpoints/integrations/integrations";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { PlugSocketIcon } from "@hugeicons/core-free-icons";
import { useCopilotModal } from "../../useCopilotModal";
import { ConnectorRow } from "./ConnectorRow";
import { InputsSection } from "./InputsSection";
import { McpConnectorRow } from "./McpConnectorRow";
import { QuestionsSection } from "./QuestionsSection";
import {
  toConnectorRows,
  type ConnectorRequest,
  type InputsRequest,
  type McpConnectorRequest,
  type QuestionRequest,
} from "./helpers";

interface Props {
  connectors: ConnectorRequest[];
  mcp: McpConnectorRequest[];
  inputs: InputsRequest[];
  questions: QuestionRequest[];
  isReady: boolean;
  onProceed: () => void;
}

/** Everything the chain still needs from the user, stacked below it as one
 *  card per kind of ask — connectors, run inputs, questions. The questions
 *  card carries its own Skip/Add footer (Add drafts the combined reply into
 *  the chat input); an inputs-only stack falls back to a lone Proceed, and
 *  a connectors-only card has no button: connecting is the whole ask. */
export function ChainActionCard({
  connectors,
  mcp,
  inputs,
  questions,
  isReady,
  onProceed,
}: Props) {
  const { openModal } = useCopilotModal();
  const providersQuery = useGetV1ListProviders({
    query: {
      select: (response) => (response.status === 200 ? response.data : []),
    },
  });

  const rows = toConnectorRows(connectors, providersQuery.data ?? []);
  // Two tool calls against the same MCP server ask once — connecting it
  // satisfies both hidden cards via their live-credential re-check.
  const mcpRows = mcp.filter(
    (request, i) =>
      mcp.findIndex((other) => other.serverUrl === request.serverUrl) === i,
  );
  const hasInputs = inputs.some(
    (request) => request.schema !== null || request.hasAdvanced,
  );
  const hasQuestions = questions.some(
    (request) => request.questions.length > 0,
  );
  // The questions card's send gates on its own answers only — an unready
  // sibling (e.g. an unconnected MCP row) must not freeze it.
  const questionsReady =
    hasQuestions &&
    questions.every((request) =>
      request.questions.every(
        (q) => (request.answers[q.keyword] ?? "").trim().length > 0,
      ),
    );
  if (rows.length === 0 && mcp.length === 0 && !hasInputs && !hasQuestions)
    return null;

  return (
    <div className="my-2 flex flex-col gap-2">
      {(rows.length > 0 || mcp.length > 0) && (
        <div className="overflow-hidden rounded-3xl border border-zinc-100 bg-white">
          <div className="flex items-center gap-2.5 border-b border-zinc-100 px-4 py-3">
            <Icon icon={PlugSocketIcon} size={18} className="text-zinc-400" />
            <span className="text-sm font-medium text-zinc-900">
              Plug in what this needs
            </span>
          </div>
          {rows.map((row) => (
            <ConnectorRow key={row.provider} row={row} />
          ))}
          {mcpRows.map((request) => (
            <McpConnectorRow key={request.id} request={request} />
          ))}
          <div className="border-t border-zinc-100 px-4 py-3">
            <span className="text-sm text-zinc-500">
              Looking for something else?{" "}
              <button
                type="button"
                onClick={() => openModal("integrations")}
                className="font-medium text-zinc-900 underline underline-offset-2"
              >
                Browse all connectors
              </button>
            </span>
          </div>
        </div>
      )}

      <InputsSection requests={inputs} />

      {hasQuestions && (
        <div className="overflow-hidden rounded-3xl border border-zinc-100 bg-white">
          <QuestionsSection
            requests={questions}
            isReady={questionsReady}
            onProceed={onProceed}
          />
        </div>
      )}

      {hasInputs && !hasQuestions && (
        <Button
          variant="primary"
          size="small"
          className="w-fit"
          disabled={!isReady}
          onClick={onProceed}
        >
          Proceed
        </Button>
      )}
    </div>
  );
}
