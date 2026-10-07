import { beforeEach, describe, expect, it, vi } from "vitest";
import { putV2RecordCredentialPicksForThisChat } from "@/app/api/__generated__/endpoints/chat/chat";
import { buildCredentialSelections, reportCredentialPicks } from "../helpers";

vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => ({
  putV2RecordCredentialPicksForThisChat: vi.fn(),
}));

const record = vi.mocked(putV2RecordCredentialPicksForThisChat);

const picked = {
  credentials: { id: "cred-personal", provider: "github" },
  linear_credentials: { id: "cred-linear", provider: "linear" },
  unfilled: undefined,
  half_filled: { provider: "slack" },
};

describe("buildCredentialSelections", () => {
  it("maps each chosen credential to its provider and skips unfilled fields", () => {
    expect(buildCredentialSelections(picked)).toEqual({
      github: "cred-personal",
      linear: "cred-linear",
    });
  });
});

describe("reportCredentialPicks", () => {
  beforeEach(() => record.mockReset());

  it("tells the backend which accounts were chosen for this chat", async () => {
    await reportCredentialPicks("session-1", picked);

    expect(record).toHaveBeenCalledWith("session-1", {
      selections: { github: "cred-personal", linear: "cred-linear" },
    });
  });

  it("stays quiet when there is no session or nothing was chosen", async () => {
    await reportCredentialPicks(null, picked);
    await reportCredentialPicks("session-1", { unfilled: undefined });

    expect(record).not.toHaveBeenCalled();
  });

  it("does not block the reply when recording fails", async () => {
    // With nothing recorded the backend asks again instead of guessing.
    record.mockRejectedValueOnce(new Error("network"));

    await expect(
      reportCredentialPicks("session-1", picked),
    ).resolves.toBeUndefined();
  });
});
