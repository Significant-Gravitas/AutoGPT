import { getGetV2ListChatConnectionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { getGetV1ListCredentialsMockHandler200 } from "@/app/api/__generated__/endpoints/integrations/integrations.msw";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it } from "vitest";

import { IntegrationsPanel } from "../IntegrationsPanel";

describe("IntegrationsPanel", () => {
  it("separates the subscriptions that pay for a chat from the tools a chat uses", async () => {
    server.use(
      getGetV2ListChatConnectionsMockHandler200({ offers: [] }),
      getGetV1ListCredentialsMockHandler200([]),
    );

    render(<IntegrationsPanel />);

    const headings = await screen.findAllByRole("heading", { level: 2 });
    expect(headings.map((heading) => heading.textContent)).toEqual([
      "AI subscriptions",
      "Tools your agents use",
    ]);
  });

  it("keeps the tools heading while the list is still loading", async () => {
    // The heading names what the region is, so it should not appear only once
    // the region has content -- the AI section above it behaves the same way.
    server.use(getGetV2ListChatConnectionsMockHandler200({ offers: [] }));

    render(<IntegrationsPanel />);

    expect(
      await screen.findByRole("heading", { name: "Tools your agents use" }),
    ).toBeDefined();
  });
});
