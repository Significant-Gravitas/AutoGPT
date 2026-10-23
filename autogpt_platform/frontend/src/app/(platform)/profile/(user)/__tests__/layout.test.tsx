import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const { usePathnameMock } = vi.hoisted(() => ({
  usePathnameMock: vi.fn(() => "/profile/credits"),
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

describe("legacy /profile layout", () => {
  it.each(["/profile/dashboard", "/profile/credits", "/profile/api-keys"])(
    "renders nothing on %s, which redirects to /settings",
    (path) => {
      usePathnameMock.mockReturnValue(path);

      const { container } = render(
        <Layout>
          <p>legacy page</p>
        </Layout>,
      );

      expect(container.innerHTML).toBe("");
    },
  );

  it.each(["/profile/settings", "/profile/oauth-apps"])(
    "still renders the old shell on %s, which keeps its page",
    (path) => {
      usePathnameMock.mockReturnValue(path);

      const { getByText, getAllByText } = render(
        <Layout>
          <p>legacy page</p>
        </Layout>,
      );

      expect(getByText("legacy page")).toBeDefined();
      expect(getAllByText("Integrations").length).toBeGreaterThan(0);
    },
  );
});
