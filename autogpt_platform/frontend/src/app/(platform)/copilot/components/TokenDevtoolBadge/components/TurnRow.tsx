import {
  formatTokenCount,
  turnInputTokens,
  type TokenTurn,
} from "../../../tokenDevtool/tokenMath";

interface Props {
  index: number;
  turn: TokenTurn;
}

export function TurnRow({ index, turn }: Props) {
  return (
    <div className="flex items-baseline gap-2 font-mono text-xs">
      <span className="w-6 shrink-0 text-foreground-tertiary">
        #{index + 1}
      </span>
      {turn.compacted && (
        <span className="text-warning">
          <span aria-hidden>⟲</span>
          <span className="sr-only">transcript summarized this turn</span>
        </span>
      )}
      <span className="text-foreground-emphasis">
        in {formatTokenCount(turnInputTokens(turn))}
      </span>
      <span className="text-foreground-tertiary">
        out {formatTokenCount(turn.completionTokens)}
      </span>
      <span className="ml-auto text-foreground-subtle">
        w {formatTokenCount(turn.cacheCreationTokens)}
      </span>
    </div>
  );
}
