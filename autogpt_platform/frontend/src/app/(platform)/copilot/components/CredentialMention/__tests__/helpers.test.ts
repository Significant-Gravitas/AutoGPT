import { describe, expect, it } from "vitest";
import {
  parseCredentialMentions,
  serializeCredentialMention,
  credentialMentionDisplayText,
} from "../helpers";

const account = {
  credentialId: "account-work",
  provider: "google",
  name: "Work [Gmail]",
};

describe("credential references", () => {
  it("round trips the exact account ID and escaped label", () => {
    const token = serializeCredentialMention(account);
    expect(token).toContain("credential://google/account-work");
    expect(parseCredentialMentions(`Check ${token} now`)).toEqual([
      "Check ",
      { ...account, token },
      " now",
    ]);
  });

  it("keeps identical names associated with different accounts", () => {
    expect(serializeCredentialMention(account)).not.toBe(
      serializeCredentialMention({
        ...account,
        credentialId: "account-personal",
      }),
    );
  });

  it("copies visible labels without exposing IDs", () => {
    expect(
      credentialMentionDisplayText(
        `Check ${serializeCredentialMention(account)}`,
      ),
    ).toBe("Check Work [Gmail]");
  });

  it("treats malformed encoding as text", () => {
    expect(parseCredentialMentions("[Work](credential://google/%ZZ)")).toEqual([
      "[Work](credential://google/%ZZ)",
    ]);
  });
});
