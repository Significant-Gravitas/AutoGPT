import {
  getListExpertPodsMockHandler,
  getListExpertsMockHandler,
  getUninstallExpertWorkflowMockHandler,
  getUninstallExpertWorkflowMockHandler404,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getGetV2ListLibraryAgentsMockHandler200,
  getGetV2ListLibraryAgentsResponseMock200,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV1ListExecutionSchedulesForAUserMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { Expert } from "@/app/api/__generated__/models/expert";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import TeamPage from "../page";

const toastMock = vi.hoisted(() => vi.fn());
const { setFlagStatusMock } = vi.hoisted(() => ({
  setFlagStatusMock: vi.fn(() => ({ enabled: true, ready: true })),
}));

vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return { ...actual, toast: toastMock };
});

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useFlagStatus: (flag: string) =>
      flag === "hire-experts"
        ? setFlagStatusMock()
        : actual.useFlagStatus(flag as never),
  };
});

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: () => "/team",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
  notFound: () => {
    throw new Error("NEXT_NOT_FOUND");
  },
}));

const hiredMaria: Expert = {
  id: "expert-maria",
  name: "Maria",
  avatar_url: null,
  role: "Marketing Strategist",
  bio: null,
  skills: [],
  tagline: "Grows your brand while you sleep",
  identity: "You are Maria, a senior marketing strategist.",
  voice_preferences: "Warm, concise, and direct.",
  boundaries: "Never invent customer evidence.",
  protected_soul_rules: [],
  is_template: false,
  source_template_id: "template-maria",
  is_archived: false,
  workflows: [
    {
      id: "wf-1",
      store_listing_version_id: "slv-1",
      library_agent_id: "lib-1",
      graph_id: "graph-1",
      name: "Content Calendar",
      description: null,
    },
    {
      id: "wf-2",
      store_listing_version_id: "slv-2",
      library_agent_id: "lib-2",
      graph_id: "graph-2",
      name: "SEO Audit",
      description: null,
    },
  ],
} as unknown as Expert;

function emptyLibrary() {
  const base = getGetV2ListLibraryAgentsResponseMock200();
  return {
    ...base,
    agents: [],
    pagination: {
      ...base.pagination,
      total_items: 0,
      current_page: 1,
      page_size: 100,
      total_pages: 0,
    },
  };
}

beforeEach(() => {
  server.use(
    getGetV1ListExecutionSchedulesForAUserMockHandler([]),
    getListExpertPodsMockHandler([]),
    getGetV2ListLibraryAgentsMockHandler200(emptyLibrary()),
    getListExpertsMockHandler([hiredMaria]),
  );
});

afterEach(() => {
  setFlagStatusMock.mockReturnValue({ enabled: true, ready: true });
  toastMock.mockReset();
});

async function openRemoveDialog(user: ReturnType<typeof userEvent.setup>) {
  const group = await screen.findByRole("region", { name: "Maria runs" });
  await user.click(
    within(group).getByRole("button", {
      name: "Remove Content Calendar from Maria",
    }),
  );
}

describe("TeamPage — removing a workflow", () => {
  test("confirms before uninstalling, then reports success", async () => {
    const user = userEvent.setup();
    let deletedPath: string | undefined;
    server.use(
      getUninstallExpertWorkflowMockHandler(({ request }) => {
        deletedPath = new URL(request.url).pathname;
      }),
    );

    render(<TeamPage />);
    await openRemoveDialog(user);

    expect(
      await screen.findByText("Remove Content Calendar?"),
    ).toBeDefined();
    await user.click(screen.getByTestId("remove-workflow-confirm"));

    await waitFor(() =>
      expect(deletedPath).toBe(
        "/api/proxy/api/experts/expert-maria/workflows/wf-1",
      ),
    );
    expect(toastMock).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Content Calendar removed from Maria",
      }),
    );
    await waitFor(() =>
      expect(screen.queryByText("Remove Content Calendar?")).toBeNull(),
    );
  });

  test("cancelling leaves the workflow installed", async () => {
    const user = userEvent.setup();
    let requested = false;
    server.use(
      getUninstallExpertWorkflowMockHandler(() => {
        requested = true;
      }),
    );

    render(<TeamPage />);
    await openRemoveDialog(user);
    await user.click(await screen.findByRole("button", { name: "Cancel" }));

    await waitFor(() =>
      expect(screen.queryByText("Remove Content Calendar?")).toBeNull(),
    );
    expect(requested).toBe(false);
    const group = await screen.findByRole("region", { name: "Maria runs" });
    expect(within(group).getByText("Content Calendar")).toBeDefined();
  });

  test("keeps the dialog open and warns when the request fails", async () => {
    const user = userEvent.setup();
    server.use(getUninstallExpertWorkflowMockHandler404());

    render(<TeamPage />);
    await openRemoveDialog(user);
    await user.click(await screen.findByTestId("remove-workflow-confirm"));

    await waitFor(() =>
      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Could not remove Content Calendar",
          variant: "destructive",
        }),
      ),
    );
    expect(screen.getByText("Remove Content Calendar?")).toBeDefined();
  });
});
