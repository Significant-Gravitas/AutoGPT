import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { CHANGELOG_INDEX_MD_URL, STORAGE_KEY } from "../changelog-constants";
import { ChangelogPopup } from "../ChangelogPopup";

const LATEST = "v0-6-63";
const INDEX_MD = `
| Date | Highlights |
| ---- | ---------- |
| [May 7 – June 10](${LATEST}.md) | Copilot upgrades, new blocks |
`;

beforeEach(() => {
  window.localStorage.clear();
  server.use(
    http.get(CHANGELOG_INDEX_MD_URL, () => HttpResponse.text(INDEX_MD)),
  );
});

afterEach(() => {
  server.resetHandlers();
});

describe("ChangelogPopup", () => {
  it("slides in for an unseen release with highlights and a View changelog link", async () => {
    render(<ChangelogPopup />);

    expect(
      await screen.findByText("Copilot upgrades", {}, { timeout: 3000 }),
    ).toBeDefined();
    expect(screen.getByText("new blocks")).toBeDefined();

    const link = screen.getByRole("link", { name: /view changelog/i });
    expect(link.getAttribute("href")).toContain(`/changelog/${LATEST}`);
    expect(link.getAttribute("target")).toBe("_blank");
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

  it("stays hidden for a release the user has already seen", async () => {
    window.localStorage.setItem(STORAGE_KEY, LATEST);
    render(<ChangelogPopup />);

    await new Promise((resolve) => setTimeout(resolve, 1800));
    expect(screen.queryByText("Copilot upgrades")).toBeNull();
  });
});
