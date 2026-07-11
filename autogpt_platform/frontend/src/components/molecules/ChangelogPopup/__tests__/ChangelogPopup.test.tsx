import { server } from "@/mocks/mock-server";
import { useGetFlag } from "@/services/feature-flags/use-get-flag";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CHANGELOG_INDEX_MD_URL } from "../changelog-constants";
import { ChangelogPopup } from "../ChangelogPopup";

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: vi.fn() };
});

// usePlatformChrome (which drives the gate) reads the session to decide the
// logged-out marketplace tour sidebar; a logged-in user keeps that off so the
// new-layout gate is what's under test here.
vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: () => ({ isLoggedIn: true, isUserLoading: false }),
}));

const LATEST = "v0-6-63";
const INDEX_MD = `
| Release | Highlights |
| --- | --- |
| [May 7 – June 10, 2026](/docs/platform/changelog/changelog/${LATEST}.md) | Copilot upgrades |
`;

let indexRequests = 0;

beforeEach(() => {
  indexRequests = 0;
  window.localStorage.clear();
  server.use(
    http.get(CHANGELOG_INDEX_MD_URL, () => {
      indexRequests += 1;
      return HttpResponse.text("");
    }),
  );
});

afterEach(() => {
  server.resetHandlers();
  vi.mocked(useGetFlag).mockReset();
});

describe("ChangelogPopup gating", () => {
  it("keeps the toast hidden once the new sidebar layout is active", async () => {
    vi.mocked(useGetFlag).mockReturnValue(true);
    server.use(
      http.get(CHANGELOG_INDEX_MD_URL, () => HttpResponse.text(INDEX_MD)),
    );
    render(<ChangelogPopup />);

    // Wait past the reveal delay — the toast must never surface on the new
    // layout, where the sidebar owns the changelog.
    await new Promise((resolve) => setTimeout(resolve, 1800));
    expect(screen.queryByText("Copilot upgrades")).toBeNull();
  });

  it("mounts the floating toast on the classic layout", async () => {
    vi.mocked(useGetFlag).mockReturnValue(false);
    render(<ChangelogPopup />);

    await waitFor(() => expect(indexRequests).toBeGreaterThan(0));
  });
});

describe("ChangelogPopup toast (classic layout)", () => {
  beforeEach(() => {
    vi.mocked(useGetFlag).mockReturnValue(false);
    server.use(
      http.get(CHANGELOG_INDEX_MD_URL, () => HttpResponse.text(INDEX_MD)),
    );
  });

  it("slides in for an unseen release and opens the full modal via Read more", async () => {
    render(<ChangelogPopup />);

    const readMore = await screen.findByText(
      /read more/i,
      {},
      { timeout: 3000 },
    );
    expect(screen.getByText("Copilot upgrades")).toBeDefined();

    await userEvent.click(readMore);
    expect(await screen.findByText(/view all on docs/i)).toBeDefined();
  });

  it("can be dismissed", async () => {
    render(<ChangelogPopup />);

    const dismiss = await screen.findByLabelText(
      "Dismiss changelog",
      {},
      { timeout: 3000 },
    );
    await userEvent.click(dismiss);

    await waitFor(() =>
      expect(screen.queryByText("Copilot upgrades")).toBeNull(),
    );
  });
});
