import type { UIDataTypes, UIMessage, UITools } from "ai";
import { useState } from "react";
import { carryRowKeys, type RowKeys } from "./stableRowKeys";

type ChatMessage = UIMessage<unknown, UIDataTypes, UITools>;

const NO_KEYS: RowKeys = new Map();

/**
 * The rendered rows' React keys, carried across updates that replace message
 * ids (see `carryRowKeys`). The keys are worked out again only when the ids
 * change, so a streamed token costs no extra render.
 */
export function useStableRowKeys(rows: ChatMessage[]): RowKeys {
  const ids = idsOf(rows);
  const [state, setState] = useState({ ids, rows, keys: NO_KEYS });
  if (state.ids === ids) return state.keys;
  const next = { ids, rows, keys: carryRowKeys(state.rows, state.keys, rows) };
  setState(next);
  return next.keys;
}

function idsOf(rows: ChatMessage[]) {
  return rows.map((row) => row.id).join("\u0000");
}
