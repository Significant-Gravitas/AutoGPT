import type { ExpertKickoffMetadata } from "../expertKickoff";

const RESEND_KIND = "held_follow_up_resend";

/** A Retry of a follow-up the last turn left unread, named by its id. */
export interface ResendMetadata {
  kind: typeof RESEND_KIND;
  pendingId: string;
}

export type SendMetadata = ExpertKickoffMetadata | ResendMetadata;

export function resendMetadata(pendingId: string): ResendMetadata {
  return { kind: RESEND_KIND, pendingId };
}

export function getResendPendingId(metadata: unknown): string | null {
  if (typeof metadata !== "object" || metadata === null) return null;
  const { kind, pendingId } = metadata as Partial<ResendMetadata>;
  return kind === RESEND_KIND && typeof pendingId === "string"
    ? pendingId
    : null;
}
