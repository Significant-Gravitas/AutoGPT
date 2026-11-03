"use client";

import { useGetV1ListProviders } from "@/app/api/__generated__/endpoints/integrations/integrations";
import { Button } from "@/components/atoms/Button/Button";
import { isAnswered } from "../../tools/clarifying-questions";
import { ConnectionsSection } from "./ConnectionsSection";
import { InputsSection } from "./InputsSection";
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
  /** A card opted out of auto-sending, so the stack needs a Proceed even with
   *  no inputs of its own. */
  manualProceed: boolean;
  isReady: boolean;
  onProceed: () => void;
}

/** Everything the chain still needs from the user, stacked below it as one
 *  card per kind of ask — connectors, run inputs, questions. The questions
 *  card carries its own Skip/Send footer (Send posts the combined reply as
 *  one message); an inputs-only stack falls back to a lone Proceed, and
 *  a connectors-only card has no button: connecting is the whole ask. */
export function ChainActionCard({
  connectors,
  mcp,
  inputs,
  questions,
  manualProceed,
  isReady,
  onProceed,
}: Props) {
  const providersQuery = useGetV1ListProviders({
    query: {
      select: (response) => (response.status === 200 ? response.data : []),
    },
  });

  const rows = toConnectorRows(connectors, providersQuery.data ?? []);
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
      request.questions.every((q) => isAnswered(request.answers[q.keyword])),
    );
  if (rows.length === 0 && mcp.length === 0 && !hasInputs && !hasQuestions)
    return null;

  return (
    <div className="my-2 flex flex-col gap-2">
      {(rows.length > 0 || mcp.length > 0) && (
        <ConnectionsSection rows={rows} mcp={mcp} />
      )}

      <InputsSection requests={inputs} />

      {hasQuestions && (
        <div className="overflow-hidden rounded-3xl border border-separator bg-surface">
          <QuestionsSection
            requests={questions}
            isReady={questionsReady}
            onProceed={onProceed}
          />
        </div>
      )}

      {(hasInputs || manualProceed) && !hasQuestions && (
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
