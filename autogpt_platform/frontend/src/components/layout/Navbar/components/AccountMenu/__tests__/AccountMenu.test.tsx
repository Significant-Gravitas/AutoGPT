import { IconType } from "@/components/__legacy__/ui/icons";
import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, test, vi } from "vitest";
import { AccountMenu } from "../AccountMenu";
import { MenuItemGroup } from "../../../helpers";

vi.mock("next/link", () => ({
  __esModule: true,
  default: ({ children, href, ...props }: any) => (
    <a href={href} {...props}>
      {children}
    </a>
  ),
  useLinkStatus: () => ({ pending: false }),
}));

vi.mock("@/components/molecules/Popover/Popover", () => {
  function Popover({ children }: { children: React.ReactNode }) {
    return <div>{children}</div>;
  }
  function PopoverTrigger({ children }: { children: React.ReactNode }) {
    return <div>{children}</div>;
  }
  function PopoverContent({ children }: { children: React.ReactNode }) {
    return <div>{children}</div>;
  }
  return { Popover, PopoverTrigger, PopoverContent };
});

const baseGroups: MenuItemGroup[] = [
  {
    items: [
      {
        icon: IconType.Edit,
        text: "Profile",
        href: "/settings/profile",
      },
      {
        icon: IconType.Settings,
        text: "Settings",
        href: "/settings/account",
      },
      {
        icon: IconType.Help,
        text: "Help & Docs",
        href: "https://agpt.co/docs",
        external: true,
      },
    ],
  },
  {
    items: [
      {
        icon: IconType.UploadCloud,
        text: "Publish an agent",
        onClick: vi.fn(),
      },
    ],
  },
  {
    items: [
      {
        icon: IconType.LogOut,
        text: "Log out",
      },
    ],
  },
];

describe("AccountMenu", () => {
  test("renders user name and email when loaded", () => {
    render(
      <AccountMenu
        userName="Ada Lovelace"
        userEmail="ada@example.com"
        menuItemGroups={baseGroups}
      />,
    );

    expect(screen.getByText("Ada Lovelace")).toBeDefined();
    expect(screen.getByTestId("account-menu-user-email").textContent).toBe(
      "ada@example.com",
    );
  });

  test("renders skeleton loaders when loading", () => {
    const { container } = render(
      <AccountMenu
        userName="Ada"
        userEmail="ada@example.com"
        menuItemGroups={baseGroups}
        isLoading
      />,
    );

    expect(screen.queryByText("Ada")).toBeNull();
    expect(container.querySelectorAll(".animate-pulse").length).toBeGreaterThan(
      0,
    );
  });

  test("renders skeleton when userName/email are missing", () => {
    render(<AccountMenu menuItemGroups={baseGroups} />);
    expect(screen.queryByTestId("account-menu-user-email")).toBeNull();
  });

  test("renders all menu items including links, buttons and logout", () => {
    render(
      <AccountMenu
        userName="Ada"
        userEmail="ada@example.com"
        menuItemGroups={baseGroups}
      />,
    );

    expect(screen.getByText("Profile")).toBeDefined();
    expect(screen.getByText("Settings")).toBeDefined();
    expect(screen.getByText("Help & Docs")).toBeDefined();
    expect(screen.getByText("Publish an agent")).toBeDefined();
    expect(screen.getByText("Log out")).toBeDefined();
  });

  test("renders external link with target=_blank", () => {
    render(
      <AccountMenu
        userName="Ada"
        userEmail="ada@example.com"
        menuItemGroups={baseGroups}
      />,
    );

    const helpLink = screen.getByText("Help & Docs").closest("a");
    expect(helpLink?.getAttribute("target")).toBe("_blank");
    expect(helpLink?.getAttribute("rel")).toBe("noopener noreferrer");
  });

  test("renders profile trigger with avatar fallback", () => {
    render(
      <AccountMenu
        userName="Ada"
        userEmail="ada@example.com"
        menuItemGroups={baseGroups}
      />,
    );

    expect(screen.getByTestId("profile-popout-menu-trigger")).toBeDefined();
    expect(screen.getAllByText("A").length).toBeGreaterThan(0);
  });
});
