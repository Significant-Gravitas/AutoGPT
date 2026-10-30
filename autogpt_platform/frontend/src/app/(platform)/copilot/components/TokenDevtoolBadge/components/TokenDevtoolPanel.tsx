import {
  AUTOCOMPACT_TOKENS,
  MODEL_CONTEXT_WINDOW,
  formatTokenCount,
} from "../../../tokenDevtool/tokenMath";
import type { useTokenDevtoolBadge } from "../useTokenDevtoolBadge";
import { BreakdownSection } from "./BreakdownSection";
import { ContextBar } from "./ContextBar";
import { TurnRow } from "./TurnRow";
import { Text } from "@/components/atoms/Text/Text";

type Props = ReturnType<typeof useTokenDevtoolBadge>;

export function TokenDevtoolPanel({ turns, breakdown, context }: Props) {
  return (
    <>
      <div className="flex items-baseline justify-between pb-2">
        <Text variant="body-medium-tight" as="span" tone="emphasis">
          Context window
        </Text>
        <Text variant="small-mono-tight" as="span" tone="muted">
          {context === null ? "—" : `~${formatTokenCount(context)}`} /{" "}
          {formatTokenCount(MODEL_CONTEXT_WINDOW)}
        </Text>
      </div>
      <ContextBar context={context ?? 0} />
      <p className="pt-1 font-mono text-tiny leading-inherit text-foreground-subtle">
        assumes a {formatTokenCount(MODEL_CONTEXT_WINDOW)} window; the backend
        threshold is configurable
      </p>
      <p className="pt-0.5 pb-2.5 text-right font-mono text-xs text-warning">
        summarizes ~{formatTokenCount(AUTOCOMPACT_TOKENS)}
      </p>

      {breakdown && <BreakdownSection breakdown={breakdown} />}

      {!turns?.length ? (
        <p className="text-sm text-foreground-tertiary">
          Live per-turn data starts with your next message.
        </p>
      ) : (
        <div className="flex max-h-40 flex-col gap-1 overflow-y-auto">
          {turns.map((turn, i) => (
            <TurnRow key={`${turn.at}-${i}`} index={i} turn={turn} />
          ))}
        </div>
      )}
    </>
  );
}
