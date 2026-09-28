"use client";

import { ChainRowBody } from "./ChainRowBody";
import {
  HANDOFF_TOOLS,
  HandoffApprovalNode,
  isHandoffApprovalRow,
} from "./HandoffApprovalNode";
import { HandoffRowView } from "./HandoffRowView";
import type { ChainRow } from "./helpers";

interface Props {
  row: ChainRow;
  isLast: boolean;
  readOnly?: boolean;
}

export function ChainRowView({ row, isLast, readOnly = false }: Props) {
  if (!readOnly && row.held && isHandoffApprovalRow(row.tool, row.held)) {
    return (
      <HandoffApprovalNode held={row.held} input={row.input} isLast={isLast} />
    );
  }
  if (row.tool && HANDOFF_TOOLS.has(row.tool)) {
    return <HandoffRowView row={row} isLast={isLast} readOnly={readOnly} />;
  }
  return <ChainRowBody row={row} isLast={isLast} readOnly={readOnly} />;
}
