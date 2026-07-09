import { SidebarProvider } from "@/components/ui/sidebar";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CHANGELOG_INDEX_MD_URL, STORAGE_KEY } from "../changelog-constants";
import { SidebarChangelog } from "../SidebarChangelog";

type StubModalProps = {
  selectedEntry: { slug: string } | null;
  onClose: () => void;
};

vi.mock("../components/ChangelogModal", () => ({
  ChangelogModal: ({ selectedEntry, onClose }: StubModalProps) => (
    <div data-testid="changelog-modal">
      <span>modal:{selectedEntry?.slug}</span>
      <button onClick={onClose}>close-modal</button>
    </div>
  ),
}));

const LATEST_SLUG = "v0-6-63";
const INDEX_MD = `
| Release | Highlights |
| --- | --- |
| [May 7 – June 10, 2026](https://agpt.co/docs/platform/changelog/changelog/${LATEST_SLUG}) | Copilot upgrades |
| [Apr 1 – May 6, 2026](https://agpt.co/docs/platform/changelog/changelog/v0-6-58) | Marketplace redesign |
`;

function renderChangelog() {
  return render(
    <SidebarProvider>
      <SidebarChangelog />
    </SidebarProvider>,
  );
}

beforeEach(() => {
  window.localStorage.clear();
  server.use(
    http.get(CHANGELOG_INDEX_MD_URL, () => HttpResponse.text(INDEX_MD)),
  );
});

afterEach(() => {
  server.resetHandlers();
});

describe("SidebarChangelog", () => {
  it("renders the What's New entry", () => {
    renderChangelog();
    expect(screen.getByRole("button", { name: /what's new/i })).toBeDefined();
  });

  it("shows the unseen indicator when the latest release is new", async () => {
    renderChangelog();
    expect(await screen.findByTestId("changelog-unseen-dot")).toBeDefined();
  });

  it("hides the unseen indicator when the latest release was already seen", async () => {
    window.localStorage.setItem(STORAGE_KEY, LATEST_SLUG);
    renderChangelog();

    // Let the index resolve, then confirm no dot was shown.
    await screen.findByRole("button", { name: /what's new/i });
    await waitFor(() => {
      expect(screen.queryByTestId("changelog-unseen-dot")).toBeNull();
    });
  });

  it("opens the modal and marks the latest release seen on click", async () => {
    renderChangelog();

    // Wait for entries to load (dot present) before opening.
    await screen.findByTestId("changelog-unseen-dot");
    await userEvent.click(screen.getByRole("button", { name: /what's new/i }));

    expect(await screen.findByTestId("changelog-modal")).toBeDefined();
    expect(screen.getByText(`modal:${LATEST_SLUG}`)).toBeDefined();
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe(LATEST_SLUG);
    expect(screen.queryByTestId("changelog-unseen-dot")).toBeNull();
  });
});
