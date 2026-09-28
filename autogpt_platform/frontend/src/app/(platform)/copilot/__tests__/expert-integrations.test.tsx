import { getListExpertCredentialsMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { ExpertCredentialRef } from "@/app/api/__generated__/models/expertCredentialRef";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  within,
} from "@/tests/integrations/test-utils";
import { describe, expect, it, vi } from "vitest";
import { ThreadHeader } from "../components/ChatMessagesContainer/components/ThreadHeader";

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: () => true };
});

const mariaIdentity = {
  id: "expert-maria",
  name: "Maria",
  avatarUrl: null,
  role: "Marketing Strategist",
  isArchived: false,
  readOnlyReason: null,
};

function credential(provider: string): ExpertCredentialRef {
  return {
    credential_id: `cred-${provider}`,
    provider,
    title: `${provider} account`,
    type: "oauth2",
  };
}

function renderHeader() {
  return render(<ThreadHeader expertIdentity={mariaIdentity} />);
}

describe("expert integrations under the thread chip", () => {
  it("lists every integration by name", async () => {
    server.use(
      getListExpertCredentialsMockHandler([
        credential("linkedin"),
        credential("notion"),
        credential("github"),
        credential("slack"),
      ]),
    );

    renderHeader();

    const list = await screen.findByTestId("expert-integrations");
    expect(within(list).getAllByRole("listitem")).toHaveLength(4);
    expect(within(list).getByText("slack account")).toBeDefined();
    expect(within(list).getByText("linkedin account")).toBeDefined();
  });

  it("keeps integrations and actions out of the chip itself", async () => {
    server.use(getListExpertCredentialsMockHandler([credential("linkedin")]));

    renderHeader();

    await screen.findByTestId("expert-integrations");
    const chip = screen.getByLabelText(/^Maria — /);
    expect(within(chip).queryByRole("img", { name: "LinkedIn" })).toBeNull();
    expect(
      within(screen.getByTestId("expert-thread-header")).queryByRole("button"),
    ).toBeNull();
  });

  it("names an MCP integration after the service, not its URL", async () => {
    server.use(
      getListExpertCredentialsMockHandler([
        {
          credential_id: "cred-mcp",
          provider: "mcp",
          title: "MCP: mcp.sentry.dev",
          type: "host_scoped",
        },
      ]),
    );

    renderHeader();

    expect(await screen.findByText("Sentry")).toBeDefined();
    expect(screen.queryByText("MCP: mcp.sentry.dev")).toBeNull();
  });

  it("keeps the integration's name when its logo fails to load", async () => {
    server.use(getListExpertCredentialsMockHandler([credential("linkedin")]));

    renderHeader();

    const list = await screen.findByTestId("expert-integrations");
    const logo = within(list).getByRole("img", { name: "LinkedIn" });
    fireEvent.error(logo);

    // The PNG is missing for plenty of providers, so the fallback glyph must
    // still announce which integration it stands for.
    expect(within(list).getByRole("img", { name: "LinkedIn" })).toBeDefined();
  });

  it("renders nothing when the expert reaches no integrations", async () => {
    server.use(getListExpertCredentialsMockHandler([]));

    renderHeader();

    expect(await screen.findByTestId("expert-thread-header")).toBeDefined();
    expect(screen.queryByTestId("expert-integrations")).toBeNull();
  });
});
