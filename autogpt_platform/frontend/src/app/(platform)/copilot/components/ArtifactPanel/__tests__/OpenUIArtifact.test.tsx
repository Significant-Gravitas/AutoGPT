import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { server } from "@/mocks/mock-server";
import { formView } from "@/lib/openui/__tests__/capability-fixtures";
import { OPENUI_MIME } from "@/lib/openui/file";
import { registerChatSender } from "../../../helpers/sendToChat";
import { useCopilotUIStore } from "../../../store";
import { openUIArtifactRef } from "../../OpenUIFileView/helpers";
import { ArtifactPanel } from "../ArtifactPanel";
import { classifyArtifact } from "../helpers";

const router = vi.hoisted(() => ({ push: vi.fn() }));
vi.mock("next/navigation", () => ({
  useRouter: () => ({ ...router, replace: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/copilot",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

const source = formView(
  'Field("notes", "Notes", "Original", "", false)',
).replace("[form])", '[form], "panel-review")');

function serveFile(viewState: Record<string, unknown> | "missing") {
  server.use(
    http.get("/api/proxy/api/workspace/files/view-1/download", () =>
      HttpResponse.text(source),
    ),
    http.get("*/api/workspace/files/view-1/view-state", () =>
      viewState === "missing"
        ? HttpResponse.json({ detail: "Not found" }, { status: 404 })
        : HttpResponse.json(viewState),
    ),
  );
}

const savedView = {
  file_id: "view-1",
  view_id: "panel-review",
  source,
  checksum: "c1",
  state: {},
  revision: 1,
  origin_session_id: "home-chat",
};

function openView(
  mimeType: string | null = OPENUI_MIME,
  title = "Trip.openui",
) {
  useCopilotUIStore
    .getState()
    .openArtifact({ ...openUIArtifactRef("view-1", title), mimeType });
}

describe("saved views in the side panel", () => {
  beforeEach(() => {
    localStorage.clear();
    router.push.mockClear();
  });
  afterEach(() => {
    useCopilotUIStore.getState().resetArtifactPanel();
  });

  it("classifies .openui files and the view MIME as interactive views", () => {
    expect(classifyArtifact(null, "plan.openui").type).toBe("openui");
    expect(classifyArtifact(OPENUI_MIME, "plan").type).toBe("openui");
    expect(classifyArtifact(OPENUI_MIME, "plan").hasSourceToggle).toBe(true);
  });

  it("renders a saved view from its file and shows its source on request", async () => {
    serveFile(savedView);
    openView();
    render(<ArtifactPanel />);

    expect(await screen.findByLabelText("Notes")).toBeDefined();
    expect(screen.getByText("· Interactive view")).toBeDefined();

    fireEvent.click(screen.getByRole("button", { name: "Source" }));
    expect(await screen.findByText(/root = Workspace\("Review"/)).toBeDefined();
    expect(screen.queryByLabelText("Notes")).toBeNull();
  });

  it("opens a .openui file by extension even without the view MIME", async () => {
    serveFile(savedView);
    openView("text/plain");
    render(<ArtifactPanel />);
    expect(await screen.findByLabelText("Notes")).toBeDefined();
  });

  it("falls back to the source as text when the file is not a saved view", async () => {
    serveFile("missing");
    openView();
    render(<ArtifactPanel />);

    expect(await screen.findByText(/root = Workspace\("Review"/)).toBeDefined();
    expect(screen.queryByLabelText("Notes")).toBeNull();
  });

  it("sends a submit outside the home chat to that chat and opens it", async () => {
    serveFile(savedView);
    openView();
    render(<ArtifactPanel />);

    await screen.findByLabelText("Notes");
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() =>
      expect(router.push).toHaveBeenCalledWith("/copilot?sessionId=home-chat"),
    );
    const homeSend = vi.fn();
    const unregister = registerChatSender("home-chat", homeSend);
    expect(homeSend.mock.calls[0][0]).toContain("Use my choices");
    expect(await screen.findByText("Sent to the conversation")).toBeDefined();
    unregister();
  });

  it("shows only the saved source on a public share", async () => {
    let viewStateReads = 0;
    server.use(
      http.get("/api/proxy/api/workspace/files/view-1/download", () =>
        HttpResponse.text(source),
      ),
      http.get("*/api/workspace/files/view-1/view-state", () => {
        viewStateReads++;
        return HttpResponse.json(savedView);
      }),
    );
    openView();
    render(<ArtifactPanel readOnly />);

    const notes = (await screen.findByLabelText("Notes")) as HTMLInputElement;
    expect(notes.disabled).toBe(true);
    expect(
      screen.getByRole("button", { name: "Continue" }).hasAttribute("disabled"),
    ).toBe(true);
    expect(viewStateReads).toBe(0);
  });
});
