import { getPostV2StartSessionDesktopMockHandler } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { server } from "@/mocks/mock-server";
import { capturePostHogEvent } from "@/services/analytics/posthog-capture";
import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useCopilotUIStore } from "../../../store";
import { CopilotChatActionsContext } from "../../CopilotChatActionsProvider/useCopilotChatActions";
import { DesktopStreamCard } from "../DesktopStreamCard";

vi.mock("@/services/analytics/posthog-capture", () => ({
  capturePostHogEvent: vi.fn(),
}));

let mobile = false;
vi.mock("../../../useIsMobile", () => ({
  useIsMobile: () => mobile,
}));

const STREAM = {
  kind: "desktop_stream",
  url: "/api/proxy/api/desktop-preview?token=abc",
  provider: "e2b",
  sandbox_id: "sbx-1",
};

const OWNED = { ...STREAM, requires_auth: true };
const LIVE_STREAM = "https://6080-sbx-1.e2b.app/vnc.html?password=pw";

function previewAnswers(isServed: (token: string | null) => boolean) {
  return http.get("/api/proxy/api/desktop-preview", ({ request }) =>
    isServed(new URL(request.url).searchParams.get("token"))
      ? new HttpResponse(null, {
          status: 307,
          headers: { Location: LIVE_STREAM },
        })
      : HttpResponse.json({ detail: "Desktop stream ended" }, { status: 410 }),
  );
}

afterEach(() => {
  vi.useRealTimers();
});

beforeEach(() => {
  vi.mocked(capturePostHogEvent).mockClear();
  mobile = false;
  useCopilotUIStore.getState().resetArtifactPanel();
  useCopilotUIStore.getState().closeArtifactPanel({ persist: false });
});

describe("DesktopStreamCard", () => {
  it("opens the computer face for a desktop the model just started", () => {
    render(<DesktopStreamCard stream={STREAM} />);

    const panel = useCopilotUIStore.getState().artifactPanel;
    expect(panel.computer?.sandbox_id).toBe("sbx-1");
    expect(panel.isComputerOpen).toBe(true);
  });

  it("on mobile remembers the desktop but opens no face the sheet cannot show", () => {
    mobile = true;
    render(<DesktopStreamCard stream={STREAM} />);

    const panel = useCopilotUIStore.getState().artifactPanel;
    expect(panel.computer?.sandbox_id).toBe("sbx-1");
    expect(panel.isComputerOpen).toBe(false);
    expect(panel.isOpen).toBe(false);
  });

  it("registers nothing for a shared transcript", () => {
    render(<DesktopStreamCard stream={STREAM} readOnly />);

    expect(useCopilotUIStore.getState().artifactPanel.computer).toBeNull();
  });

  it("replaces a stream that stops on screen with a notice, and counts it once", async () => {
    vi.useFakeTimers({
      toFake: ["setTimeout", "clearTimeout", "Date"],
      shouldAdvanceTime: true,
    });
    let served = true;
    server.use(previewAnswers(() => served));
    render(<DesktopStreamCard stream={OWNED} />);

    expect(
      await screen.findByTitle("Interactive desktop (sbx-1)"),
    ).toBeDefined();
    expect(capturePostHogEvent).not.toHaveBeenCalled();

    // The turn ends and the box pauses: the link stops redirecting.
    served = false;
    await vi.advanceTimersByTimeAsync(10_000);

    expect(
      await screen.findByText("This live desktop has stopped."),
    ).toBeDefined();
    expect(screen.queryByTitle(/Interactive desktop/)).toBeNull();
    expect(capturePostHogEvent).toHaveBeenCalledTimes(1);
    expect(capturePostHogEvent).toHaveBeenCalledWith("desktop_stream_ended", {
      seconds_live: 10,
    });
  });

  it("reconnects a stream that had already stopped, without counting it", async () => {
    server.use(
      previewAnswers((token) => token === "fresh"),
      getPostV2StartSessionDesktopMockHandler({
        kind: "desktop_stream",
        url: "/api/proxy/api/desktop-preview?token=fresh",
        provider: "e2b",
        sandbox_id: "sbx-1",
        requires_auth: true,
      }),
    );
    render(
      <CopilotChatActionsContext.Provider
        value={{ onSend: vi.fn(), chatSurface: "copilot", sessionId: "s-1" }}
      >
        <DesktopStreamCard stream={OWNED} />
      </CopilotChatActionsContext.Provider>,
    );

    await userEvent.click(
      await screen.findByRole("button", { name: "Reconnect" }),
    );

    const frame = await screen.findByTitle("Interactive desktop (sbx-1)");
    expect(frame.getAttribute("src")).toBe(
      "/api/proxy/api/desktop-preview?token=fresh",
    );
    // The side panel follows the card onto the fresh stream.
    expect(useCopilotUIStore.getState().artifactPanel.computer?.url).toBe(
      "/api/proxy/api/desktop-preview?token=fresh",
    );
    expect(capturePostHogEvent).not.toHaveBeenCalled();
  });
});
