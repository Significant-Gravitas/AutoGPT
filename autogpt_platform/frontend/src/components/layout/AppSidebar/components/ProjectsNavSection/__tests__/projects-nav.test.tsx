import {
  getGetV2ListBoardsMockHandler200,
  getGetV2ListSessionsMockHandler200,
  getPostV2CreateBoardMockHandler200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import type { BoardSummary } from "@/app/api/__generated__/models/boardSummary";
import { SidebarProvider } from "@/components/ui/sidebar";
import { server } from "@/mocks/mock-server";
import { Flag } from "@/services/feature-flags/use-get-flag";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppSidebar } from "../../../AppSidebar";

const nav = vi.hoisted(() => ({ pathname: "/home", push: vi.fn() }));
const toastSpy = vi.hoisted(() => vi.fn());
const flags = vi.hoisted(() => ({ taskBoard: true }));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "user-1", email: "alice@example.com", role: "user" },
    isLoggedIn: true,
    isUserLoading: false,
  }),
}));

vi.mock("next/link", () => ({
  __esModule: true,
  default: ({
    children,
    href,
    ...props
  }: {
    children: ReactNode;
    href: string;
  }) => (
    <a href={href} {...props}>
      {children}
    </a>
  ),
  useLinkStatus: () => ({ pending: false }),
}));

vi.mock("next/navigation", async (importOriginal) => {
  const actual = await importOriginal<typeof import("next/navigation")>();
  return {
    ...actual,
    useRouter: () => ({ push: nav.push, replace: vi.fn(), prefetch: vi.fn() }),
    usePathname: () => nav.pathname,
    useSearchParams: () => new URLSearchParams(),
  };
});

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: Flag) =>
      flag === actual.Flag.TASK_BOARD ? flags.taskBoard : false,
  };
});

vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return {
    ...actual,
    toast: (...args: Parameters<typeof actual.toast>) => toastSpy(...args),
  };
});

const HOME_ID = "0a0b0c0d-1e2f-4a3b-8c4d-5e6f7a8b9c0d";
const B1 = "b1a2c3d4-5e6f-4a7b-8c9d-0e1f2a3b4c5d";
const B2 = "b2b3c4d5-6e7f-4a8b-9c0d-1e2f3a4b5c6d";
const B_NEW = "c3c4d5e6-7f8a-4b9c-8d0e-2f3a4b5c6d7e";

function board(id: string, name: string, position: number): BoardSummary {
  return { id, name, is_default: position === 0, position };
}

const HOME = board(HOME_ID, "Home", 0);

let boardLists = 0;
const originalScrollTo = Element.prototype.scrollTo;

function serveBoards(boards: () => BoardSummary[]) {
  server.use(
    getGetV2ListBoardsMockHandler200(() => {
      boardLists += 1;
      return { boards: boards() };
    }),
  );
}

function renderSidebar({ open = true } = {}) {
  return render(
    <SidebarProvider defaultOpen={open}>
      <AppSidebar />
    </SidebarProvider>,
  );
}

async function findProjectsGroup() {
  const heading = await screen.findByRole("button", {
    name: "Projects",
    expanded: true,
  });
  return heading.closest<HTMLElement>('[data-sidebar="group"]')!;
}

function getNewProjectButton() {
  return screen.getByRole("button", { name: "New project" });
}

beforeEach(() => {
  nav.pathname = "/home";
  nav.push.mockReset();
  toastSpy.mockReset();
  flags.taskBoard = true;
  boardLists = 0;
  server.use(getGetV2ListSessionsMockHandler200({ sessions: [], total: 0 }));
  // happy-dom has no Web Animations or scrolling; the rail jump needs both.
  Element.prototype.getAnimations = () => [];
  Element.prototype.scrollTo = vi.fn();
});

afterEach(() => {
  Element.prototype.scrollTo = originalScrollTo;
  Reflect.deleteProperty(Element.prototype, "getAnimations");
});

describe("Projects in the sidebar", () => {
  it("lists the projects in order, without Home", async () => {
    serveBoards(() => [HOME, board(B1, "Launch", 1), board(B2, "Research", 2)]);
    renderSidebar();

    const group = await findProjectsGroup();
    await within(group).findByRole("link", { name: "Launch" });
    const links = within(group).getAllByRole("link");
    expect(links.map((link) => link.textContent)).toEqual([
      "Launch",
      "Research",
    ]);
    expect(links.map((link) => link.getAttribute("href"))).toEqual([
      `/boards/${B1}`,
      `/boards/${B2}`,
    ]);
    expect(within(group).queryByText("Home")).toBeNull();
    expect(
      within(group).getByRole("button", { name: "New project" }),
    ).toBeDefined();
  });

  it("marks the project being viewed", async () => {
    nav.pathname = `/boards/${B1}`;
    serveBoards(() => [HOME, board(B1, "Launch", 1), board(B2, "Research", 2)]);
    renderSidebar();

    const group = await findProjectsGroup();
    const launch = await within(group).findByRole("link", { name: "Launch" });
    expect(launch.getAttribute("data-active")).toBe("true");
    expect(
      within(group)
        .getByRole("link", { name: "Research" })
        .getAttribute("data-active"),
    ).toBe("false");
  });

  it("creates the next untitled project and opens it to name it", async () => {
    const created: unknown[] = [];
    let boards = [HOME, board(B1, "Untitled project", 1)];
    serveBoards(() => boards);
    server.use(
      getPostV2CreateBoardMockHandler200(async ({ request }) => {
        const body = (await request.json()) as { name: string };
        created.push(body);
        const next = board(B_NEW, body.name, 2);
        boards = [...boards, next];
        return next;
      }),
    );
    renderSidebar();
    const group = await findProjectsGroup();
    await within(group).findByRole("link", { name: "Untitled project" });

    await userEvent.click(getNewProjectButton());

    await waitFor(() =>
      expect(nav.push).toHaveBeenCalledWith(`/boards/${B_NEW}?rename=1`),
    );
    expect(created).toEqual([{ name: "Untitled project 2" }]);
    expect(
      await within(group).findByRole("link", { name: "Untitled project 2" }),
    ).toBeDefined();
    expect(boardLists).toBeGreaterThan(1);
  });

  it("says why a project was refused and adds no row", async () => {
    let posts = 0;
    serveBoards(() => [HOME, board(B1, "Launch", 1)]);
    server.use(
      http.post("*/api/chat/boards", () => {
        posts += 1;
        return HttpResponse.json(
          { detail: "You can have up to 20 projects" },
          { status: 422 },
        );
      }),
    );
    renderSidebar();
    const group = await findProjectsGroup();
    await within(group).findByRole("link", { name: "Launch" });

    await userEvent.click(getNewProjectButton());

    await waitFor(() =>
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          description: "You can have up to 20 projects",
          variant: "destructive",
        }),
      ),
    );
    expect(posts).toBe(1);
    expect(within(group).getAllByRole("link")).toHaveLength(1);
    expect(nav.push).not.toHaveBeenCalled();
  });

  it("turns New project off at 20 projects and says why", async () => {
    let posts = 0;
    const projects = Array.from({ length: 19 }, (_, i) =>
      board(
        `00000000-0000-4000-8000-${String(i).padStart(12, "0")}`,
        `P${i}`,
        i + 1,
      ),
    );
    serveBoards(() => [HOME, ...projects]);
    server.use(
      http.post("*/api/chat/boards", () => {
        posts += 1;
        return HttpResponse.json(board(B_NEW, "Untitled project", 20));
      }),
    );
    renderSidebar();
    const group = await findProjectsGroup();
    await within(group).findByRole("link", { name: "P18" });

    const button = getNewProjectButton();
    expect(button.getAttribute("aria-disabled")).toBe("true");
    await userEvent.hover(button);
    expect(
      (await screen.findAllByText("You can have up to 20 projects")).length,
    ).toBeGreaterThan(0);

    await userEvent.click(button);
    expect(posts).toBe(0);
    expect(nav.push).not.toHaveBeenCalled();
  });

  it("jumps to Projects from the collapsed rail", async () => {
    nav.pathname = `/boards/${B1}`;
    serveBoards(() => [HOME, board(B1, "Launch", 1)]);
    renderSidebar({ open: false });

    const rail = screen
      .getAllByRole("button", { name: "Projects" })
      .find((button) => !button.hasAttribute("aria-expanded"))!;
    expect(rail.getAttribute("data-active")).toBe("true");

    await userEvent.click(rail);

    const sidebar = document
      .querySelector('[data-sidebar="sidebar"]')
      ?.closest("[data-state]");
    expect(sidebar?.getAttribute("data-state")).toBe("expanded");
    const heading = screen.getByRole("button", {
      name: "Projects",
      expanded: true,
    });
    await waitFor(() => expect(document.activeElement).toBe(heading));
  });

  it("is not shown without the task board", async () => {
    flags.taskBoard = false;
    serveBoards(() => [HOME, board(B1, "Launch", 1)]);
    renderSidebar();

    expect(
      await screen.findByRole("button", { name: "Recent chats" }),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: "Projects" })).toBeNull();
    expect(screen.queryByRole("button", { name: "New project" })).toBeNull();
    expect(boardLists).toBe(0);
  });
});
