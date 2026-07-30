import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { server } from "@/mocks/mock-server";
import { useOrgTeamStore } from "@/services/org-team/store";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ShareAgentDialog } from "./ShareAgentDialog";

// happy-dom can't render Radix Dialog portals — mock the molecule so the
// dialog body renders inline regardless of open/media-query state.
function MockDialog({ children }: { children: ReactNode }) {
  return <div role="dialog">{children}</div>;
}
function MockDialogContent({ children }: { children: ReactNode }) {
  return <>{children}</>;
}
function MockDialogFooter({ children }: { children: ReactNode }) {
  return <>{children}</>;
}
MockDialog.Content = MockDialogContent;
MockDialog.Footer = MockDialogFooter;
vi.mock("@/components/molecules/Dialog/Dialog", () => ({ Dialog: MockDialog }));

const TEAM_A = {
  id: "team-a",
  name: "Growth",
  slug: "growth",
  isDefault: false,
  joinPolicy: "closed",
  orgId: "org-1",
};
const TEAM_B = {
  id: "team-b",
  name: "Platform",
  slug: "platform",
  isDefault: false,
  joinPolicy: "closed",
  orgId: "org-1",
};

const AGENT = {
  graph_id: "graph-1",
  graph_version: 3,
  name: "My Agent",
  can_access_graph: true,
} as unknown as LibraryAgent;

const GRANTS_URL =
  "http://localhost:3000/api/proxy/api/orgs/org-1/graphs/graph-1/grants";

const GRANT = {
  id: "grant-1",
  agent_graph_id: "graph-1",
  agent_graph_version: 3,
  follow_latest: false,
  principal_type: "TEAM",
  principal_id: "team-a",
  capability: "EXECUTE",
  credential_mode: "CONSUMER",
  org_id: "org-1",
  created_by_user_id: "u1",
  created_at: new Date("2026-07-01T00:00:00Z").toISOString(),
};

function seedTeams(teams: (typeof TEAM_A)[] = [TEAM_A, TEAM_B]) {
  useOrgTeamStore.setState({
    activeOrgID: "org-1",
    activeTeamID: null,
    orgs: [],
    teams,
    isLoaded: true,
  });
}

function renderDialog(agent: LibraryAgent = AGENT) {
  return render(<ShareAgentDialog agent={agent} isOpen setIsOpen={() => {}} />);
}

beforeEach(() => {
  seedTeams();
});

describe("ShareAgentDialog", () => {
  it("creates a grant pinned to the current version with the chosen team and capability", async () => {
    let body: Record<string, unknown> | null = null;
    server.use(
      http.get(GRANTS_URL, () => HttpResponse.json([], { status: 200 })),
      http.post(GRANTS_URL, async (info) => {
        body = (await info.request.json()) as Record<string, unknown>;
        return HttpResponse.json(GRANT, { status: 200 });
      }),
    );

    renderDialog();

    fireEvent.click(await screen.findByRole("combobox", { name: "Team" }));
    fireEvent.click(await screen.findByRole("option", { name: "Growth" }));

    await userEvent.click(screen.getByRole("button", { name: "Share" }));

    await waitFor(() => expect(body).not.toBeNull());
    expect(body).toMatchObject({
      principal_type: "TEAM",
      principal_id: "team-a",
      capability: "EXECUTE",
      credential_mode: "CONSUMER",
      follow_latest: false,
      graph_version: 3,
    });
  });

  it("shares the latest version with owner credentials when toggled", async () => {
    let body: Record<string, unknown> | null = null;
    server.use(
      http.get(GRANTS_URL, () => HttpResponse.json([], { status: 200 })),
      http.post(GRANTS_URL, async (info) => {
        body = (await info.request.json()) as Record<string, unknown>;
        return HttpResponse.json(GRANT, { status: 200 });
      }),
    );

    renderDialog();

    fireEvent.click(await screen.findByRole("combobox", { name: "Team" }));
    fireEvent.click(await screen.findByRole("option", { name: "Platform" }));

    fireEvent.click(screen.getByRole("combobox", { name: "Access" }));
    fireEvent.click(await screen.findByRole("option", { name: "Can view" }));

    fireEvent.click(
      screen.getByRole("switch", { name: "Always share latest version" }),
    );

    fireEvent.click(screen.getByRole("combobox", { name: "Credentials" }));
    fireEvent.click(
      await screen.findByRole("option", { name: "Run with my credentials" }),
    );
    // Owner-credential warning surfaces.
    expect(screen.getByText("Runs use your connected accounts.")).toBeDefined();

    await userEvent.click(screen.getByRole("button", { name: "Share" }));

    await waitFor(() => expect(body).not.toBeNull());
    expect(body).toMatchObject({
      principal_id: "team-b",
      capability: "VIEW",
      credential_mode: "OWNER",
      follow_latest: true,
      graph_version: null,
    });
  });

  it("hides the credential mode selector for non-owners and omits credential_mode", async () => {
    let body: Record<string, unknown> | null = null;
    server.use(
      http.get(GRANTS_URL, () => HttpResponse.json([], { status: 200 })),
      http.post(GRANTS_URL, async (info) => {
        body = (await info.request.json()) as Record<string, unknown>;
        return HttpResponse.json(GRANT, { status: 200 });
      }),
    );

    renderDialog({ ...AGENT, can_access_graph: false });

    expect(screen.queryByRole("combobox", { name: "Credentials" })).toBeNull();

    fireEvent.click(await screen.findByRole("combobox", { name: "Team" }));
    fireEvent.click(await screen.findByRole("option", { name: "Growth" }));
    await userEvent.click(screen.getByRole("button", { name: "Share" }));

    await waitFor(() => expect(body).not.toBeNull());
    expect(body).not.toBeNull();
    expect(Object.prototype.hasOwnProperty.call(body, "credential_mode")).toBe(
      false,
    );
  });

  it("lists existing grants and revokes one", async () => {
    let deleted = false;
    server.use(
      http.get(GRANTS_URL, () => HttpResponse.json([GRANT], { status: 200 })),
      http.delete(`${GRANTS_URL}/grant-1`, () => {
        deleted = true;
        return new HttpResponse(null, { status: 204 });
      }),
    );

    renderDialog();

    // Row renders with the team badge and capability/version summary.
    expect(await screen.findByTestId("share-grant-row")).toBeDefined();
    expect(screen.getByText("Growth")).toBeDefined();
    expect(screen.getByText(/Can run · v3/)).toBeDefined();

    await userEvent.click(screen.getByRole("button", { name: "Revoke" }));
    await waitFor(() => expect(deleted).toBe(true));
  });
});
