import {
  getGetV2GetLibraryAgentByGraphIdMockHandler,
  getPostV2ForkLibraryAgentMockHandler,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { server } from "@/mocks/mock-server";
import { TEAM_HEADER_NAME } from "@/services/org-team/headers";
import { useOrgTeamStore } from "@/services/org-team/store";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ReadOnlyBanner } from "../ReadOnlyBanner";

vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: () => ({ user: { id: "user-owner" } }),
}));

const TEAM = {
  id: "team-a",
  name: "Growth",
  slug: "growth",
  isDefault: false,
  joinPolicy: "closed",
  orgId: "org-1",
};

const LIB_AGENT = {
  id: "lib-1",
  graph_id: "graph-1",
  name: "My Agent",
} as LibraryAgent;

function renderBanner() {
  return render(
    <NuqsTestingAdapter searchParams={{ flowID: "graph-1" }}>
      <ReadOnlyBanner />
    </NuqsTestingAdapter>,
  );
}

beforeEach(() => {
  window.localStorage.clear();
  useOrgTeamStore.setState({
    activeOrgID: "org-1",
    activeTeamID: null,
    orgs: [],
    teams: [TEAM],
    isLoaded: true,
  });
  server.use(getGetV2GetLibraryAgentByGraphIdMockHandler(() => LIB_AGENT));
});

describe("ReadOnlyBanner duplicate into team", () => {
  it("forks the agent with the picked team as the X-Team-Id header", async () => {
    let sentTeamHeader: string | null = null;
    const forkSpy = vi.fn();
    server.use(
      getPostV2ForkLibraryAgentMockHandler((info) => {
        forkSpy();
        sentTeamHeader = info.request.headers.get(TEAM_HEADER_NAME);
        return { ...LIB_AGENT, id: "lib-2", graph_id: "graph-2" };
      }),
    );

    renderBanner();

    const combo = await screen.findByRole("combobox", {
      name: "Duplicate into team",
    });
    fireEvent.click(combo);
    fireEvent.click(await screen.findByRole("option", { name: "Growth" }));

    fireEvent.click(screen.getByRole("button", { name: /Duplicate/ }));

    await waitFor(() => expect(forkSpy).toHaveBeenCalledTimes(1));
    expect(sentTeamHeader).toBe(TEAM.id);
  });

  it("forks without a team header when Organization is kept", async () => {
    let sentTeamHeader: string | null = "unset";
    const forkSpy = vi.fn();
    server.use(
      getPostV2ForkLibraryAgentMockHandler((info) => {
        forkSpy();
        sentTeamHeader = info.request.headers.get(TEAM_HEADER_NAME);
        return { ...LIB_AGENT, id: "lib-2", graph_id: "graph-2" };
      }),
    );

    renderBanner();

    const duplicateBtn = await screen.findByRole("button", {
      name: /Duplicate/,
    });
    await waitFor(() =>
      expect(duplicateBtn.hasAttribute("disabled")).toBe(false),
    );
    fireEvent.click(duplicateBtn);

    await waitFor(() => expect(forkSpy).toHaveBeenCalledTimes(1));
    expect(sentTeamHeader).toBeNull();
  });
});
