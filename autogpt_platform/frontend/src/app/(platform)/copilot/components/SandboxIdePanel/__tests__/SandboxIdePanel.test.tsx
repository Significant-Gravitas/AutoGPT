import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { useCopilotUIStore } from "../../../store";
import { SandboxIdePanel } from "../SandboxIdePanel";

const { flagMock, toastMock, treePathSpy, writeSpy } = vi.hoisted(() => ({
  flagMock: vi.fn(() => true),
  toastMock: vi.fn(),
  treePathSpy: vi.fn(),
  writeSpy: vi.fn(),
}));

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: { COPILOT_SANDBOX_IDE: "copilot-sandbox-ide" },
  useGetFlag: () => flagMock(),
}));

vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return { ...actual, toast: (...args: unknown[]) => toastMock(...args) };
});

vi.mock("@xterm/xterm", () => ({ Terminal: vi.fn() }));
vi.mock("@xterm/addon-fit", () => ({ FitAddon: vi.fn() }));
vi.mock("@/lib/supabase/actions", () => ({
  getWebSocketToken: vi.fn(async () => ({ token: "t" })),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/copilot",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";

function Harness({ sessionId = "s1" }: { sessionId?: string }) {
  const enabled = useGetFlag(Flag.COPILOT_SANDBOX_IDE);
  return enabled ? <SandboxIdePanel sessionId={sessionId} /> : null;
}

function treeHandler() {
  return http.get(
    "*/api/chat/sessions/:sessionId/sandbox/tree",
    ({ request }) => {
      const path = new URL(request.url).searchParams.get("path") ?? "";
      treePathSpy(path);
      if (path === "src") {
        return HttpResponse.json({
          entries: [{ name: "b.py", path: "src/b.py", type: "file" }],
        });
      }
      return HttpResponse.json({
        entries: [
          { name: "src", path: "src", type: "dir" },
          { name: "a.py", path: "a.py", type: "file" },
        ],
      });
    },
  );
}

function changesHandler(files: { path: string; status: string }[]) {
  return http.get("*/api/chat/sessions/:sessionId/sandbox/changes", () =>
    HttpResponse.json({ is_git_repo: files.length > 0, files }),
  );
}

function fileHandler() {
  return http.get("*/api/chat/sessions/:sessionId/sandbox/file", ({ request }) => {
    const path = new URL(request.url).searchParams.get("path") ?? "";
    return HttpResponse.json({ path, content: "print('hi')", truncated: false });
  });
}

function writeHandler() {
  return http.put(
    "*/api/chat/sessions/:sessionId/sandbox/file",
    async ({ request }) => {
      const body = (await request.json()) as { path: string; content: string };
      writeSpy(body);
      return HttpResponse.json({ ...body, truncated: false });
    },
  );
}

beforeEach(() => {
  flagMock.mockReturnValue(true);
  toastMock.mockClear();
  treePathSpy.mockClear();
  writeSpy.mockClear();
  const store = useCopilotUIStore.getState();
  store.openSandboxIde();
  store.setSandboxIdeTab("files");
  store.selectSandboxFile(null);
  server.use(treeHandler(), changesHandler([]), fileHandler(), writeHandler());
});

describe("SandboxIdePanel", () => {
  test("renders nothing when the flag is off", () => {
    flagMock.mockReturnValue(false);
    render(<Harness />);
    expect(screen.queryByText("Sandbox")).toBeNull();
  });

  test("renders root tree entries and lazy-loads a directory on expand", async () => {
    render(<Harness />);
    expect(await screen.findByText("a.py")).toBeTruthy();
    const dir = await screen.findByText("src");

    fireEvent.click(dir);
    await waitFor(() => expect(treePathSpy).toHaveBeenCalledWith("src"));
    expect(await screen.findByText("b.py")).toBeTruthy();
  });

  test("clicking a file requests its content", async () => {
    const fileSpy = vi.fn();
    server.use(
      http.get("*/api/chat/sessions/:sessionId/sandbox/file", ({ request }) => {
        const path = new URL(request.url).searchParams.get("path") ?? "";
        fileSpy(path);
        return HttpResponse.json({
          path,
          content: "print('hi')",
          truncated: false,
        });
      }),
    );
    render(<Harness />);
    fireEvent.click(await screen.findByText("a.py"));
    await waitFor(() => expect(fileSpy).toHaveBeenCalledWith("a.py"));
  });

  test("changes tab lists changed files, and shows empty state when none", async () => {
    server.use(changesHandler([{ path: "a.py", status: "M" }]));
    render(<Harness />);
    fireEvent.click(screen.getByRole("tab", { name: /changes/i }));
    expect(await screen.findByText("a.py")).toBeTruthy();

    server.use(changesHandler([]));
    useCopilotUIStore.getState().setSandboxIdeTab("files");
    useCopilotUIStore.getState().setSandboxIdeTab("changes");
    expect(await screen.findByText("No changes yet")).toBeTruthy();
  });

  test("Ctrl+S saves the edited file and toasts success", async () => {
    const { container } = render(<Harness />);
    fireEvent.click(await screen.findByText("a.py"));

    const editor = await waitFor(() => {
      const el = container.querySelector(".cm-editor");
      if (!el) throw new Error("editor not mounted");
      return el;
    });

    fireEvent.keyDown(editor, { key: "s", ctrlKey: true });
    await waitFor(() => expect(writeSpy).toHaveBeenCalled());
    expect(writeSpy.mock.calls[0][0]).toMatchObject({ path: "a.py" });
    await waitFor(() =>
      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({ variant: "success" }),
      ),
    );
  });
});
