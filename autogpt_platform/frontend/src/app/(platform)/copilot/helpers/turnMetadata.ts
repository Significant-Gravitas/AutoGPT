/** Client-side timing metadata recorded per assistant turn. */
export interface TurnMetadata {
  messageId: string;
  durationMs: number;
}

/** Map of messageId -> TurnMetadata */
export type TurnMetadataMap = Map<string, TurnMetadata>;
