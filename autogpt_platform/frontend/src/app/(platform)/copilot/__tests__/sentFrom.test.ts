import { describe, expect, it } from "vitest";
import {
  getSentFromDisplayName,
  getSentFromMetadata,
  getSessionSentFrom,
  isSessionOpeningMessage,
} from "../sentFrom";

const SESSION_ID = "3f2c1a9e-7b4d-4e8a-9c1b-2d3e4f5a6b7c";
const EXPERT_ID = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d";

describe("getSentFromMetadata", () => {
  it("reads the sending session and expert off message metadata", () => {
    expect(
      getSentFromMetadata({
        from_session_id: SESSION_ID,
        from_expert_id: EXPERT_ID,
        from_expert_name: "Ari",
      }),
    ).toEqual({
      sessionId: SESSION_ID,
      expertId: EXPERT_ID,
      expertName: "Ari",
    });
  });

  it("keeps a plain Otto sender as a null expert", () => {
    expect(
      getSentFromMetadata({
        from_session_id: SESSION_ID,
        from_expert_id: null,
      }),
    ).toEqual({ sessionId: SESSION_ID, expertId: null, expertName: null });
  });

  it("ignores metadata without a valid sending session id", () => {
    expect(getSentFromMetadata(null)).toBeNull();
    expect(getSentFromMetadata({ kind: "expert_kickoff" })).toBeNull();
    expect(getSentFromMetadata({ from_session_id: "not-a-uuid" })).toBeNull();
  });
});

describe("getSessionSentFrom", () => {
  it("falls back to the session's delegation provenance", () => {
    expect(
      getSessionSentFrom({
        delegated_by_session_id: SESSION_ID,
        delegated_by_expert_id: EXPERT_ID,
      }),
    ).toEqual({ sessionId: SESSION_ID, expertId: EXPERT_ID, expertName: null });
  });

  it("returns null for a thread nobody delegated", () => {
    expect(getSessionSentFrom({})).toBeNull();
    expect(getSessionSentFrom(undefined)).toBeNull();
  });
});

describe("getSentFromDisplayName", () => {
  it("prefers the name stamped on the message, then the roster, then a default", () => {
    const stamped = {
      sessionId: SESSION_ID,
      expertId: EXPERT_ID,
      expertName: "Ari",
    };
    expect(getSentFromDisplayName(stamped, "Roster Ari")).toBe("Ari");
    const unnamed = { ...stamped, expertName: null };
    expect(getSentFromDisplayName(unnamed, "Roster Ari")).toBe("Roster Ari");
    expect(getSentFromDisplayName(unnamed, null)).toBe("an expert");
    expect(getSentFromDisplayName({ ...unnamed, expertId: null }, null)).toBe(
      "Otto",
    );
  });
});

describe("isSessionOpeningMessage", () => {
  function message(id: string, role: "user" | "assistant" = "user") {
    return { id, role, parts: [] };
  }

  it("is the user row hydrated at DB sequence 0", () => {
    expect(isSessionOpeningMessage(message("sess-seq-0"))).toBe(true);
  });

  it("is never a later row, whatever the client retained", () => {
    expect(isSessionOpeningMessage(message("sess-seq-1"))).toBe(false);
    expect(isSessionOpeningMessage(message("sess-seq-40"))).toBe(false);
  });

  it("is never a row without a DB sequence", () => {
    expect(isSessionOpeningMessage(message("user-streamed"))).toBe(false);
    expect(isSessionOpeningMessage(message("sess-idx-0"))).toBe(false);
  });

  it("is never an assistant row", () => {
    expect(isSessionOpeningMessage(message("sess-seq-0", "assistant"))).toBe(
      false,
    );
  });
});
