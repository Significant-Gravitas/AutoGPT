import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import {
  afterAll,
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import { MobileWarning } from "../MobileWarning";

const mockUseBreakpoint = vi.fn();

vi.mock("@/lib/hooks/useBreakpoint", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/lib/hooks/useBreakpoint")>();
  return {
    ...actual,
    useBreakpoint: () => mockUseBreakpoint(),
  };
});

describe("MobileWarning", () => {
  const originalSetPointerCapture = HTMLElement.prototype.setPointerCapture;
  const originalReleasePointerCapture =
    HTMLElement.prototype.releasePointerCapture;
  const originalHasPointerCapture = HTMLElement.prototype.hasPointerCapture;

  beforeAll(() => {
    HTMLElement.prototype.setPointerCapture = vi.fn();
    HTMLElement.prototype.releasePointerCapture = vi.fn();
    HTMLElement.prototype.hasPointerCapture = vi.fn(() => false);
  });

  afterAll(() => {
    HTMLElement.prototype.setPointerCapture = originalSetPointerCapture;
    HTMLElement.prototype.releasePointerCapture = originalReleasePointerCapture;
    HTMLElement.prototype.hasPointerCapture = originalHasPointerCapture;
  });

  beforeEach(() => {
    mockUseBreakpoint.mockReturnValue("sm");
  });

  afterEach(() => {
    mockUseBreakpoint.mockReset();
  });

  it("renders the warning on mobile breakpoints", async () => {
    render(<MobileWarning />);
    expect(
      await screen.findByText(/builder works best on desktop/i),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: /continue anyway/i }),
    ).toBeDefined();
  });

  it("does not render the warning on large breakpoints", () => {
    mockUseBreakpoint.mockReturnValue("lg");
    render(<MobileWarning />);
    expect(screen.queryByText(/builder works best on desktop/i)).toBeNull();
    expect(
      screen.queryByRole("button", { name: /continue anyway/i }),
    ).toBeNull();
  });

  it("dismisses when the user clicks 'Continue anyway'", async () => {
    render(<MobileWarning />);
    const dialog = await screen.findByRole("dialog");
    fireEvent.click(
      await screen.findByRole("button", { name: /continue anyway/i }),
    );
    await waitFor(() => {
      expect(dialog.getAttribute("data-state")).toBe("closed");
    });
  });
});
