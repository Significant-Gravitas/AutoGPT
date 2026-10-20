import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const { usePathnameMock } = vi.hoisted(() => ({
  usePathnameMock: vi.fn(() => "/profile"),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn() }),
  usePathname: usePathnameMock,
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useGetFlag: () => true,
}));

import Layout from "../layout";
import UserProfilePage from "../page";
import UserIntegrationsPage from "../integrations/page";

describe("legacy /profile layout", () => {
  it.each([
    ["/profile", UserProfilePage],
    ["/profile/integrations", UserIntegrationsPage],
  ])("renders nothing on %s, which redirects to /settings", (path, Page) => {
    usePathnameMock.mockReturnValue(path);

    expect(
      render(
        <Layout>
          <Page />
        </Layout>,
      ).container.innerHTML,
    ).toBe("");
    // The page is a placeholder the layout never mounts; content added here would never show.
    expect(render(<Page />).container.innerHTML).toBe("");
  });

  it("still renders the old shell on /profile/settings, which keeps its page", () => {
    usePathnameMock.mockReturnValue("/profile/settings");

    const { getByText, getAllByText } = render(
      <Layout>
        <p>settings page</p>
      </Layout>,
    );

    expect(getByText("settings page")).toBeDefined();
    expect(getAllByText("Integrations").length).toBeGreaterThan(0);
  });
});
