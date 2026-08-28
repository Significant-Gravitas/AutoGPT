import { getListExpertCredentialsMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { ExpertCredentialRef } from "@/app/api/__generated__/models/expertCredentialRef";
import { server } from "@/mocks/mock-server";
import { render, screen, within } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
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
  return render(
    <ThreadHeader
      expertIdentity={mariaIdentity}
      readOnly={false}
      areFilesOpen={false}
    />,
  );
}

describe("expert integrations in the thread header", () => {
  it("shows the first three logos and counts the rest", async () => {
    server.use(
      getListExpertCredentialsMockHandler([
        credential("linkedin"),
        credential("notion"),
        credential("github"),
        credential("slack"),
        credential("gmail"),
      ]),
    );

    renderHeader();

    const cluster = await screen.findByTestId("expert-integrations");
    expect(within(cluster).getAllByRole("img")).toHaveLength(3);
    expect(within(cluster).getByText("+2")).toBeDefined();
  });

  it("omits the counter when everything fits", async () => {
    server.use(
      getListExpertCredentialsMockHandler([
        credential("linkedin"),
        credential("notion"),
      ]),
    );

    renderHeader();

    const cluster = await screen.findByTestId("expert-integrations");
    expect(within(cluster).getAllByRole("img")).toHaveLength(2);
    expect(within(cluster).queryByText(/^\+/)).toBeNull();
  });

  it("lists every integration by name once opened", async () => {
    server.use(
      getListExpertCredentialsMockHandler([
        credential("linkedin"),
        credential("notion"),
        credential("github"),
        credential("slack"),
      ]),
    );

    renderHeader();
    await userEvent.click(await screen.findByTestId("expert-integrations"));

    expect(await screen.findByText("slack account")).toBeDefined();
    expect(screen.getByText("linkedin account")).toBeDefined();
    expect(
      screen.getByRole("link", { name: /Manage what Maria can access/ }),
    ).toBeDefined();
  });

  it("renders nothing when the expert reaches no integrations", async () => {
    server.use(getListExpertCredentialsMockHandler([]));

    renderHeader();

    expect(await screen.findByTestId("expert-thread-header")).toBeDefined();
    expect(screen.queryByTestId("expert-integrations")).toBeNull();
  });
});
