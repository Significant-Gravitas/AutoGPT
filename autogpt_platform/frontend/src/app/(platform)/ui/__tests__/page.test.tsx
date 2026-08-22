import { render, screen } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, test, vi } from "vitest";

const { getFlagMock } = vi.hoisted(() => ({
  getFlagMock: vi.fn(() => true),
}));

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: { NEW_TOOL_UI: "new-tool-ui" },
  useGetFlag: () => getFlagMock(),
}));

const notFoundMock = vi.hoisted(() => vi.fn());
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: () => "/ui",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
  notFound: () => {
    notFoundMock();
    throw new Error("NEXT_NOT_FOUND");
  },
}));

afterEach(() => {
  getFlagMock.mockReturnValue(true);
  notFoundMock.mockClear();
});

import ToolUIPreviewPage from "../page";

describe("ToolUIPreviewPage", () => {
  test("renders the tool UI preview when the flag is on", async () => {
    getFlagMock.mockReturnValue(true);

    render(<ToolUIPreviewPage />);

    expect(await screen.findByText("hire_expert · preview")).toBeDefined();
    expect(notFoundMock).not.toHaveBeenCalled();
  });

  test("calls notFound() when the flag is off", () => {
    getFlagMock.mockReturnValue(false);

    expect(() => render(<ToolUIPreviewPage />)).toThrow("NEXT_NOT_FOUND");
    expect(notFoundMock).toHaveBeenCalled();
  });
});
