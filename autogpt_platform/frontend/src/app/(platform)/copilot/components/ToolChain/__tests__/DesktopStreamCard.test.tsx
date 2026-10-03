import { render } from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useCopilotUIStore } from "../../../store";
import { DesktopStreamCard } from "../DesktopStreamCard";

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

beforeEach(() => {
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
});
