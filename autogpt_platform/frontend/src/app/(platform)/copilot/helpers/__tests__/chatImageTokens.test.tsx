import { describe, expect, it } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import { CredentialMentionMarkdown } from "../../components/CredentialMention/CredentialMentionMarkdown";
import { CHAT_IMAGE_TOKEN_PREFIXES } from "../chatImageTokens";

function TokenLabel({ alt }: React.JSX.IntrinsicElements["img"]) {
  return <span>{alt}</span>;
}

describe.each(Object.entries(CHAT_IMAGE_TOKEN_PREFIXES))(
  "the %s image token",
  (_, prefix) => {
    it("leaves the message text around it as written", async () => {
      const { container } = render(
        <CredentialMentionMarkdown components={{ img: TokenLabel }}>
          {`Check ![Work Gmail](${prefix}google/work-id) today.`}
        </CredentialMentionMarkdown>,
      );

      await screen.findByText("Work Gmail");
      expect(container.textContent).toBe("Check Work Gmail today.");
    });
  },
);
