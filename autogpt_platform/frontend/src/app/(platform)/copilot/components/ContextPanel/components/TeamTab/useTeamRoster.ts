"use client";

import { getGetV2GetSessionQueryOptions } from "@/app/api/__generated__/endpoints/chat/chat";
import type { UIMessage } from "ai";
import { useQueries } from "@tanstack/react-query";
import { useRef } from "react";
import { useCopilotStreamStore } from "../../../../copilotStreamStore";
import { useExpertMap } from "../../../../useExpertMap";
import { isSessionLive } from "../../../ToolChain/SubSessionLive";
import {
  type DelegationRow,
  foldDelegations,
  isLiveDelegationStatus,
} from "./helpers";

const POLL_MS = 3000;
const POLL_CAP_MS = 5 * 60_000;
const EMPTY_MESSAGES: UIMessage[] = [];

export interface TeamRosterRow extends DelegationRow {
  /** The frozen status corrected by the polled sub-session: a delegate
   *  output frozen at "running" flips to completed once the sub-session
   *  goes idle, or to "unknown" when the poll dies. */
  effectiveStatus: string;
}

function useEffectiveStatuses(rows: DelegationRow[]) {
  const mountedAtRef = useRef(Date.now());
  const polled = rows.filter(
    (row) =>
      row.subSessionId !== null &&
      row.toolState === "done" &&
      isLiveDelegationStatus(row.status),
  );
  const results = useQueries({
    queries: polled.map((row) =>
      getGetV2GetSessionQueryOptions(row.subSessionId!, undefined, {
        query: {
          refetchInterval: (query) => {
            if (query.state.status === "error") return false;
            if (Date.now() - mountedAtRef.current > POLL_CAP_MS) return false;
            const raw = query.state.data;
            const session = raw && raw.status === 200 ? raw.data : null;
            return !session || isSessionLive(session) ? POLL_MS : false;
          },
        },
      }),
    ),
  });
  const statuses = new Map<string, string>();
  polled.forEach((row, index) => {
    const result = results[index];
    const raw = result?.data;
    const session = raw && raw.status === 200 ? raw.data : null;
    const capped = Date.now() - mountedAtRef.current > POLL_CAP_MS;
    if (result?.isError || (capped && (!session || isSessionLive(session)))) {
      statuses.set(row.key, "unknown");
    } else if (session && !isSessionLive(session)) {
      statuses.set(row.key, "completed");
    }
  });
  return statuses;
}

export function useTeamRoster(sessionId: string | null) {
  const messages = useCopilotStreamStore((s) =>
    sessionId
      ? (s.messageSnapshots[sessionId] ?? EMPTY_MESSAGES)
      : EMPTY_MESSAGES,
  );
  const { expertsById } = useExpertMap();
  const folded = foldDelegations(messages);
  const statuses = useEffectiveStatuses(folded);

  const rows: TeamRosterRow[] = folded.map((row) => {
    const expert = row.expertId ? expertsById.get(row.expertId) : undefined;
    return {
      ...row,
      expertName: row.expertName ?? expert?.name ?? null,
      expertRole: row.expertRole ?? expert?.role ?? null,
      expertAvatarUrl: row.expertAvatarUrl ?? expert?.avatarUrl ?? null,
      effectiveStatus: statuses.get(row.key) ?? row.status,
    };
  });
  const liveCount = rows.filter(
    (row) =>
      row.toolState !== "error" && isLiveDelegationStatus(row.effectiveStatus),
  ).length;

  return {
    rows,
    liveCount,
    settledCount: rows.length - liveCount,
  };
}
