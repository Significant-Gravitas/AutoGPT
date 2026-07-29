import {
  getListExpertsMockHandler,
  getListExpertsMockHandler401,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { Expert } from "@/app/api/__generated__/models/expert";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, test, vi } from "vitest";
import TeamPage from "../page";

const { setFlagStatusMock } = vi.hoisted(() => ({
  setFlagStatusMock: vi.fn(() => ({ enabled: true, ready: true })),
}));

afterEach(() => {
  setFlagStatusMock.mockReturnValue({ enabled: true, ready: true });
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

const notFoundMock = vi.hoisted(() => vi.fn());
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
    notFoundMock();
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
};

describe("TeamPage", () => {
  test("renders the Autopilot card first", async () => {
    server.use(getListExpertsMockHandler([hiredMaria]));

    render(<TeamPage />);

    const autopilot = await screen.findByText("Autopilot");
    expect(screen.getByText(/runs the shop/i)).toBeDefined();

    const maria = await screen.findByText("Maria");
    expect(
      autopilot.compareDocumentPosition(maria) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  test("renders hired experts with workflow names and count", async () => {
    server.use(getListExpertsMockHandler([hiredMaria]));

    render(<TeamPage />);

    expect(await screen.findByText("Maria")).toBeDefined();
    expect(screen.getByText("Marketing Strategist")).toBeDefined();
    expect(screen.getByText("Content Calendar")).toBeDefined();
    expect(screen.getByText("SEO Audit")).toBeDefined();
    expect(screen.getByText("2 workflows")).toBeDefined();
  });

  test("links Chat to the expert's copilot thread", async () => {
    server.use(getListExpertsMockHandler([hiredMaria]));

    render(<TeamPage />);

    await screen.findByText("Maria");
    const chatLinks = screen.getAllByRole("link", { name: "Chat" });
    const hrefs = chatLinks.map((link) => link.getAttribute("href"));
    expect(hrefs).toContain("/copilot");
    expect(hrefs).toContain(`/copilot?expertId=${hiredMaria.id}`);

    expect(
      screen.getByRole("button", { name: "Install workflow" }),
    ).toBeDefined();
  });

  test("shows empty state linking to the marketplace when no experts are hired", async () => {
    server.use(getListExpertsMockHandler([]));

    render(<TeamPage />);

    expect(await screen.findByText("Autopilot")).toBeDefined();
    const link = await screen.findByRole("link", {
      name: "Browse the marketplace",
    });
    expect(link.getAttribute("href")).toBe("/marketplace");
  });

  test("shows an error card when loading experts fails", async () => {
    server.use(getListExpertsMockHandler401());

    render(<TeamPage />);

    expect(await screen.findByText("Something went wrong")).toBeDefined();
  });

  test("calls notFound() when the flag is resolved and disabled", () => {
    setFlagStatusMock.mockReturnValueOnce({ enabled: false, ready: true });
    notFoundMock.mockClear();

    try {
      render(<TeamPage />);
    } catch {
      // React surfaces the thrown notFound() error; the assertion below is
      // what we actually care about.
    }

    expect(notFoundMock).toHaveBeenCalled();
  });
});
