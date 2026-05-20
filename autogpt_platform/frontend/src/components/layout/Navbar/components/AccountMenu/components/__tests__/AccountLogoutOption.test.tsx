import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, test, vi } from "vitest";
import { AccountLogoutOption } from "../AccountLogoutOption";

const replaceMock = vi.fn();

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: replaceMock,
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: () => "/",
  useSearchParams: () => new URLSearchParams(),
}));

vi.mock("next/link", () => ({
  __esModule: true,
  default: ({ children, href, ...props }: any) => (
    <a href={href} {...props}>
      {children}
    </a>
  ),
  useLinkStatus: () => ({ pending: false }),
}));

describe("AccountLogoutOption", () => {
  test("renders a destructive Log out button", () => {
    render(<AccountLogoutOption />);
    const button = screen.getByRole("button", { name: /log out/i });
    expect(button.tagName).toBe("BUTTON");
    expect(button.className).toContain("hover:bg-red-50");
  });

  test("calls router.replace('/logout') when clicked", () => {
    render(<AccountLogoutOption />);
    fireEvent.click(screen.getByRole("button", { name: /log out/i }));
    expect(replaceMock).toHaveBeenCalledWith("/logout");
  });
});
