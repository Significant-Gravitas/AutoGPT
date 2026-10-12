import { getListExpertCredentialsMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getListWorkspaceFilesMockHandler200 } from "@/app/api/__generated__/endpoints/workspace/workspace.msw";
import type { ExpertCredentialRef } from "@/app/api/__generated__/models/expertCredentialRef";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  within,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ThreadHeader } from "../components/ChatMessagesContainer/components/ThreadHeader";
import { ContextPanel } from "../components/ContextPanel/ContextPanel";
import { ContextPanelToggle } from "../components/ContextPanel/ContextPanelToggle";
import { useCopilotUIStore } from "../store";

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: () => true };
});

const maria = { id: "expert-maria", name: "Maria" };

const mariaIdentity = {
  ...maria,
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

function resetPanel() {
  useCopilotUIStore.setState((s) => ({
    contextPanelExpert: null,
    artifactPanel: {
      ...s.artifactPanel,
      isOpen: false,
      activeArtifact: null,
      activeTab: "files",
      mode: "artifact",
      isComputerOpen: false,
    },
  }));
}

function renderControls() {
  return render(
    <>
      <ContextPanelToggle sessionId="session-1" expert={maria} />
      <ContextPanel sessionId="session-1" />
    </>,
  );
}

beforeEach(() => {
  server.use(
    getListWorkspaceFilesMockHandler200({
      files: [],
      offset: 0,
      has_more: false,
    }),
  );
  resetPanel();
});

afterEach(resetPanel);

describe("expert integrations in the chat controls", () => {
  it("shows the first two logos and counts the rest", async () => {
    server.use(
      getListExpertCredentialsMockHandler([
        credential("linkedin"),
        credential("notion"),
        credential("github"),
        credential("slack"),
        credential("gmail"),
      ]),
    );

    renderControls();

    const cluster = await screen.findByTestId("expert-integrations");
    expect(within(cluster).getAllByRole("img")).toHaveLength(2);
    expect(within(cluster).getByText("+3")).toBeDefined();
  });

  it("omits the counter when everything fits", async () => {
    server.use(
      getListExpertCredentialsMockHandler([
        credential("linkedin"),
        credential("notion"),
      ]),
    );

    renderControls();

    const cluster = await screen.findByTestId("expert-integrations");
    expect(within(cluster).getAllByRole("img")).toHaveLength(2);
    expect(within(cluster).queryByText(/^\+/)).toBeNull();
  });

  it("opens the side panel on the expert's integrations", async () => {
    server.use(
      getListExpertCredentialsMockHandler([
        credential("linkedin"),
        credential("notion"),
        credential("github"),
      ]),
    );

    renderControls();
    fireEvent.click(await screen.findByTestId("expert-integrations"));

    expect(await screen.findByText("Maria's Integrations")).toBeDefined();
    expect(await screen.findByText("github account")).toBeDefined();
    expect(screen.getByLabelText("Hide integrations")).toBeDefined();
    expect(useCopilotUIStore.getState().artifactPanel.activeTab).toBe(
      "integrations",
    );
  });

  it("keeps the integration's name when its logo fails to load", async () => {
    server.use(getListExpertCredentialsMockHandler([credential("linkedin")]));

    renderControls();

    const cluster = await screen.findByTestId("expert-integrations");
    fireEvent.error(within(cluster).getByRole("img", { name: "LinkedIn" }));

    // The PNG is missing for plenty of providers, so the fallback glyph must
    // still announce which integration it stands for.
    expect(
      within(cluster).getByRole("img", { name: "LinkedIn" }),
    ).toBeDefined();
  });

  it("shows a placeholder that still opens the panel when there are none", async () => {
    server.use(getListExpertCredentialsMockHandler([]));

    renderControls();

    const placeholder = await screen.findByTestId("expert-integrations-empty");
    expect(screen.queryByTestId("expert-integrations")).toBeNull();
    fireEvent.click(placeholder);

    expect(await screen.findByText("Maria's Integrations")).toBeDefined();
    expect(screen.getByLabelText("Hide integrations")).toBeDefined();
  });

  it("keeps integrations and actions out of the thread chip", async () => {
    server.use(getListExpertCredentialsMockHandler([credential("linkedin")]));

    render(<ThreadHeader expertIdentity={mariaIdentity} />);

    const header = await screen.findByTestId("expert-thread-header");
    expect(within(header).queryByRole("img", { name: "LinkedIn" })).toBeNull();
    expect(within(header).queryByRole("button")).toBeNull();
  });
});
