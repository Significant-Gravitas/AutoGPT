import type { Href } from "expo-router";

import type { BackendClient } from "../../api/backend";
import { listSessions } from "../../api/chat";
import type { ExpertIdentity } from "../../api/experts";
import { resolveAvatarURL } from "../../design/components/Avatar";
import type { ExpertDirectoryState } from "./expertDirectory";

/** Otto's portrait, as the web app shows it (`AutopilotAvatar`). */
const OTTO_AVATAR_PATH = "/autogpt-characters/v1.1/otto/neutral/128.webp";

export interface Participant {
  expertID: string | null;
  name: string;
  role: string | null;
  avatarURL: string | null;
  /** Why the conversation can't take new messages, if it can't. */
  readOnlyReason: string | null;
}

export function ottoParticipant(webOrigin: string): Participant {
  return {
    expertID: null,
    name: "Otto",
    role: "Your AI assistant",
    avatarURL: resolveAvatarURL(OTTO_AVATAR_PATH, webOrigin),
    readOnlyReason: null,
  };
}

export function expertParticipant(
  identity: ExpertIdentity | undefined,
  expertID: string,
  directory: ExpertDirectoryState["status"],
  webOrigin: string,
): Participant {
  if (!identity) {
    // Once the roster is known, an expert missing from it can't be messaged.
    const settled = directory === "ready";
    return {
      expertID,
      name: settled ? "Former expert" : "Expert",
      role: null,
      avatarURL: null,
      readOnlyReason: settled
        ? "This expert is no longer on your team, so this conversation is read-only."
        : null,
    };
  }
  return {
    expertID,
    name: identity.name,
    role: identity.jobTitle ?? identity.role,
    avatarURL: resolveAvatarURL(identity.avatarURL, webOrigin),
    readOnlyReason: identity.isArchived
      ? `You let ${identity.name} go, so this conversation is read-only.`
      : null,
  };
}

export function participantFor(
  expertID: string | null,
  directory: ExpertDirectoryState,
  webOrigin: string,
): Participant {
  if (!expertID) return ottoParticipant(webOrigin);
  return expertParticipant(
    directory.identities.get(expertID),
    expertID,
    directory.status,
    webOrigin,
  );
}

export function conversationHref(sessionID: string): Href {
  return {
    pathname: "/conversation/[sessionId]",
    params: { sessionId: sessionID },
  };
}

/** A conversation that is created on the server only when its first message is sent. */
export function newConversationHref(expertID: string | null): Href {
  return {
    pathname: "/conversation/[sessionId]",
    params: expertID
      ? { sessionId: "new", expertId: expertID }
      : { sessionId: "new" },
  };
}

/** An expert's most recent conversation, or a new one if there isn't one yet. */
export async function expertConversationHref(
  api: BackendClient,
  expertID: string,
): Promise<Href> {
  const page = await listSessions(api, {
    limit: 1,
    offset: 0,
    expertID,
    pinnedFirst: false,
  });
  const latest = page.sessions[0];
  return latest ? conversationHref(latest.id) : newConversationHref(expertID);
}
