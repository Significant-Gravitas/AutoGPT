import { getListCopilotSkillsMockHandler200 } from "@/app/api/__generated__/endpoints/skills/skills.msw";
import {
  getGetV2GetMarketplaceSkillMockHandler200,
  getGetV2ListMarketplaceSkillsMockHandler200,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import type { MarketplaceSkillDetails } from "@/app/api/__generated__/models/marketplaceSkillDetails";
import type { SkillPackageFile } from "@/app/api/__generated__/models/skillPackageFile";
import { server } from "@/mocks/mock-server";
import {
  configure,
  render,
  screen,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { beforeEach, describe, expect, test, vi } from "vitest";

import { SkillPage } from "../components/SkillPage";

// The page waits on several queries before anything renders, and CI is slower
// than a dev machine; the testing-library default is one second.
configure({ asyncUtilTimeout: 10000 });

const mockUseAuth = vi.hoisted(() => vi.fn());
const mockNotFound = vi.hoisted(() => vi.fn());
const flag = vi.hoisted(() => ({ enabled: true, ready: true }));

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    back: vi.fn(),
    forward: vi.fn(),
    prefetch: vi.fn(),
    push: vi.fn(),
    refresh: vi.fn(),
    replace: vi.fn(),
  }),
  usePathname: () => "/marketplace/skills/webapp-testing",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({ slug: SLUG }),
  notFound: mockNotFound,
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({ useAuth: mockUseAuth }));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useFlagStatus: () => flag };
});

const SLUG = "webapp-testing";
const SCRIPT_PATH = "scripts/with_server.py";
const SCRIPT_SOURCE = "#!/usr/bin/env python3\nprint('serving')\n";

// The vendored fixture's published package, byte sizes included, so the
// rendered sizes are a real package's rather than round numbers.
const PACKAGE: SkillPackageFile[] = [
  {
    path: "LICENSE.txt",
    size_bytes: 11345,
    mime_type: "text/plain",
    is_executable: false,
  },
  {
    path: "examples/console_logging.py",
    size_bytes: 1027,
    mime_type: "text/x-python",
    is_executable: false,
  },
  {
    path: "examples/element_discovery.py",
    size_bytes: 1463,
    mime_type: "text/x-python",
    is_executable: false,
  },
  {
    path: "examples/static_html_automation.py",
    size_bytes: 953,
    mime_type: "text/x-python",
    is_executable: false,
  },
  {
    path: SCRIPT_PATH,
    size_bytes: 3693,
    mime_type: "text/x-python",
    is_executable: true,
  },
];

describe("Marketplace skill package contents", () => {
  beforeEach(() => {
    flag.enabled = true;
    flag.ready = true;
    mockNotFound.mockClear();
    mockUseAuth.mockReturnValue({
      user: { id: "user-1" },
      isLoggedIn: true,
      isUserLoading: false,
    });
    server.use(
      getListCopilotSkillsMockHandler200([]),
      getGetV2ListMarketplaceSkillsMockHandler200({
        skills: [],
        pagination: {
          total_items: 0,
          total_pages: 1,
          current_page: 1,
          page_size: 3,
        },
      }),
    );
  });

  test("lists every file in the package with its size", async () => {
    server.use(detail({ files: PACKAGE }));

    render(<SkillPage slug={SLUG} />);

    const list = await screen.findByTestId("skill-package-files");
    expect(await screen.findByText("Package contents")).toBeDefined();
    for (const file of PACKAGE) {
      expect(await screen.findByText(file.path)).toBeDefined();
    }
    expect(screen.getByText("11.1 KB")).toBeDefined();
    expect(screen.getByText("3.6 KB")).toBeDefined();
    // One file in this package is executable, and the badge says which.
    const badges = within(list).getAllByText("executable");
    expect(badges.length).toBe(1);
  });

  test("shows no section at all for a single-file skill", async () => {
    server.use(detail({ files: [] }));

    render(<SkillPage slug={SLUG} />);

    await screen.findByText("Instructions");
    expect(screen.queryByTestId("skill-package-files")).toBeNull();
    expect(screen.queryByText("Package contents")).toBeNull();
  });

  test("opens a file from the list", async () => {
    server.use(detail({ files: PACKAGE }), fileText(SCRIPT_SOURCE));

    render(<SkillPage slug={SLUG} />);

    await screen.findByTestId("skill-package-files");
    await userEvent.click(screen.getByRole("button", { name: /with_server/ }));

    expect(await screen.findByText(/print\('serving'\)/)).toBeDefined();
  });

  test("opens a relative link in the body instead of following it", async () => {
    server.use(
      detail({
        files: PACKAGE,
        body: `Run the server with [the runner](./${SCRIPT_PATH}).`,
      }),
      fileText(SCRIPT_SOURCE),
    );

    render(<SkillPage slug={SLUG} />);

    const link = await screen.findByRole("button", { name: "the runner" });
    // A link the marketplace never served: it has to open the viewer, so it
    // must not be an anchor pointing at a path that 404s.
    expect(screen.queryByRole("link", { name: "the runner" })).toBeNull();
    await userEvent.click(link);

    expect(await screen.findByText(/print\('serving'\)/)).toBeDefined();
  });

  test("leaves an absolute link in the body alone", async () => {
    server.use(
      detail({
        files: PACKAGE,
        body: "See [the docs](https://example.com/scripts/with_server.py).",
      }),
    );

    render(<SkillPage slug={SLUG} />);

    const link = await screen.findByRole("link", { name: "the docs" });
    expect(link.getAttribute("href")).toBe(
      "https://example.com/scripts/with_server.py",
    );
  });

  test("says so when a file is not one the viewer can show", async () => {
    // The generated handler answers 200 whatever it is given, so the refusal
    // is mocked at the transport.
    server.use(
      detail({ files: PACKAGE }),
      http.get(`/api/proxy/api/store/skills/${SLUG}/files/*`, () =>
        HttpResponse.json({ detail: "not text" }, { status: 415 }),
      ),
    );

    render(<SkillPage slug={SLUG} />);

    await screen.findByTestId("skill-package-files");
    await userEvent.click(screen.getByRole("button", { name: /LICENSE/ }));

    expect(await screen.findByText(/can't be previewed/)).toBeDefined();
  });
});

function detail(overrides: Partial<MarketplaceSkillDetails>) {
  return getGetV2GetMarketplaceSkillMockHandler200({
    slug: SLUG,
    name: SLUG,
    title: "Webapp testing",
    description: "Test a web app end to end.",
    categories: ["development"],
    required_providers: [],
    install_count: 3,
    creator: "anthropics",
    creator_avatar: null,
    skill_listing_version_id: "version-1",
    body: "Use the scripts in this package.",
    triggers: [],
    updated_at: new Date("2026-09-01T00:00:00Z"),
    files: [],
    ...overrides,
  });
}

function fileText(text: string) {
  return http.get(`/api/proxy/api/store/skills/${SLUG}/files/*`, () =>
    HttpResponse.text(text),
  );
}
