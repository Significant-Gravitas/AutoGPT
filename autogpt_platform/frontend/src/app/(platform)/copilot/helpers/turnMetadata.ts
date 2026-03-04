/** Metadata extracted from StreamStart/StreamFinish events per message turn. */
export interface TurnMetadata {
  messageId: string;
  startedAt: string | null;
  durationMs: number | null;
}

/** Map of messageId -> TurnMetadata */
export type TurnMetadataMap = Map<string, TurnMetadata>;
