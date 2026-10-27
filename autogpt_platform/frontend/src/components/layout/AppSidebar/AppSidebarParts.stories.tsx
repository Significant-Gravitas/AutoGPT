import { getGetV2ListSessionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { getListExpertIdentitiesMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getGetFeatureFlagsMockHandler200 } from "@/app/api/__generated__/endpoints/feature-flags/feature-flags.msw";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import {
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarTrigger,
  useSidebar,
} from "@/components/molecules/Sidebar/Sidebar";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import { cn } from "@/lib/utils";
import {
  FlowIcon,
  GridViewIcon,
  Store01Icon,
} from "@hugeicons/core-free-icons";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { http, HttpResponse } from "msw";
import { NuqsAdapter } from "nuqs/adapters/react";
import type { ReactNode } from "react";
import { AppSidebar } from "./AppSidebar";
import { AppSidebarHeader } from "./components/AppSidebarHeader/AppSidebarHeader";
import {
  AppSidebarToggleProvider,
  AppSidebarTrigger,
} from "./components/AppSidebarTrigger/AppSidebarTrigger";
import { SidebarSearch } from "./components/SidebarSearch/SidebarSearch";
import { SidebarUserActions } from "./components/SidebarUserActions/SidebarUserActions";

const STORY_USER = {
  id: "story-user-1",
  email: "sam@example.com",
  role: "authenticated",
  user_metadata: { name: "Sam Rivera" },
};

const SESSIONS = [
  {
    id: "session-1",
    created_at: "2026-10-20T09:00:00Z",
    updated_at: "2026-10-23T16:20:00Z",
    title: "Weekly sales report",
    is_processing: false,
  },
  {
    id: "session-2",
    created_at: "2026-10-19T11:00:00Z",
    updated_at: "2026-10-22T10:05:00Z",
    title: "Draft the launch announcement",
    is_processing: false,
  },
  {
    id: "session-3",
    created_at: "2026-10-18T08:30:00Z",
    updated_at: "2026-10-21T14:40:00Z",
    title: "Summarise customer feedback",
    is_processing: false,
  },
];

const meta: Meta = {
  title: "Layout/AppSidebar parts",
  parameters: {
    layout: "fullscreen",
    nextjs: { navigation: { pathname: "/marketplace" } },
    msw: {
      handlers: [
        getGetV2ListSessionsMockHandler200({
          sessions: SESSIONS,
          total: SESSIONS.length,
        }),
        getListExpertIdentitiesMockHandler([]),
        getGetFeatureFlagsMockHandler200({ flags: {}, configured: false }),
        // No profile: the footer then skips the wallet, which needs the
        // onboarding provider and is hidden without billing anyway.
        http.get("*/api/proxy/api/store/profile", () =>
          HttpResponse.json({ detail: "Not found" }, { status: 404 }),
        ),
      ],
    },
  },
  beforeEach() {
    useAuthStore.setState({
      user: STORY_USER,
      hasLoadedUser: true,
      isUserLoading: false,
    });
    return () => {
      useAuthStore.setState({ user: null, hasLoadedUser: false });
    };
  },
  decorators: [
    (Story) => (
      <BackendAPIProvider>
        <NuqsAdapter>
          <TooltipProvider>
            <Story />
          </TooltipProvider>
        </NuqsAdapter>
      </BackendAPIProvider>
    ),
  ],
};

export default meta;
type Story = StoryObj;

interface ShellProps {
  open: boolean;
  children: ReactNode;
}

function Shell({ open, children }: ShellProps) {
  return (
    <SidebarProvider
      defaultOpen={open}
      className="min-h-0 flex-wrap items-start gap-8"
    >
      {children}
    </SidebarProvider>
  );
}

interface FrameProps {
  label: string;
  collapsed?: boolean;
  testID: string;
  children: ReactNode;
}

// Stands in for the <Sidebar> wrapper: the `group` element whose
// data-collapsible / data-state the parts style themselves from.
function PartFrame({ label, collapsed = false, testID, children }: FrameProps) {
  return (
    <div className="flex flex-col gap-2">
      <Text variant="body" as="p" className="text-foreground-subtle">
        {label}
      </Text>
      <div
        data-testid={testID}
        data-state={collapsed ? "collapsed" : "expanded"}
        data-collapsible={collapsed ? "icon" : ""}
        data-side="left"
        className={cn(
          "group relative flex flex-col rounded-medium border border-separator bg-sidebar text-sidebar-foreground",
          collapsed ? "w-[--sidebar-width-icon]" : "w-72",
        )}
      >
        {children}
      </div>
    </div>
  );
}

function SidebarStateReadout() {
  const { state } = useSidebar();
  return (
    <Text variant="body" as="p" className="text-foreground-subtle">
      Sidebar state: {state}
    </Text>
  );
}

function TriggerCell({
  label,
  testID,
  children,
}: {
  label: string;
  testID: string;
  children: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-2">
      <Text variant="body" as="p" className="text-foreground-subtle">
        {label}
      </Text>
      <div
        data-testid={testID}
        className="flex items-center rounded-medium border border-separator bg-surface p-2"
      >
        {children}
      </div>
    </div>
  );
}

const MENU_LINKS = [
  { name: "Workflows", icon: GridViewIcon },
  { name: "Marketplace", icon: Store01Icon },
  { name: "Build", icon: FlowIcon },
];

export const HeaderExpanded: Story = {
  render: () => (
    <Shell open>
      <PartFrame label="AppSidebarHeader, expanded" testID="header-expanded">
        <AppSidebarHeader />
      </PartFrame>
    </Shell>
  ),
};

export const HeaderCollapsed: Story = {
  render: () => (
    <Shell open={false}>
      <PartFrame
        label="AppSidebarHeader, collapsed (hover to reveal the expand button)"
        testID="header-collapsed"
        collapsed
      >
        <AppSidebarHeader />
      </PartFrame>
    </Shell>
  ),
};

export const Search: Story = {
  render: () => (
    <Shell open>
      <PartFrame label="SidebarSearch" testID="sidebar-search">
        <div className="p-2">
          <SidebarSearch />
        </div>
      </PartFrame>
    </Shell>
  ),
};

export const UserActionsExpanded: Story = {
  render: () => (
    <Shell open>
      <PartFrame label="SidebarUserActions, expanded" testID="user-actions">
        <SidebarUserActions />
      </PartFrame>
    </Shell>
  ),
};

export const UserActionsCollapsed: Story = {
  render: () => (
    <Shell open={false}>
      <PartFrame
        label="SidebarUserActions, collapsed"
        testID="user-actions-collapsed"
        collapsed
      >
        <SidebarUserActions />
      </PartFrame>
    </Shell>
  ),
};

export const Triggers: Story = {
  render: () => (
    <Shell open>
      <div className="flex flex-col gap-4">
        <SidebarStateReadout />
        <div className="flex flex-wrap gap-8">
          <TriggerCell label="SidebarTrigger" testID="sidebar-trigger">
            <SidebarTrigger />
          </TriggerCell>
          <TriggerCell label="AppSidebarTrigger" testID="app-sidebar-trigger">
            <AppSidebarToggleProvider>
              <AppSidebarTrigger />
            </AppSidebarToggleProvider>
          </TriggerCell>
        </div>
      </div>
    </Shell>
  ),
};

export const MenuButtonTooltipCollapsed: Story = {
  render: () => (
    <Shell open={false}>
      <PartFrame
        label="SidebarMenuButton with tooltip, collapsed"
        testID="menu-collapsed"
        collapsed
      >
        <SidebarMenu className="items-center p-2">
          {MENU_LINKS.map((link) => (
            <SidebarMenuItem key={link.name}>
              <SidebarMenuButton tooltip={link.name}>
                <Icon icon={link.icon} className="size-4" />
                <Text variant="body" as="span" className="truncate">
                  {link.name}
                </Text>
              </SidebarMenuButton>
            </SidebarMenuItem>
          ))}
        </SidebarMenu>
      </PartFrame>
    </Shell>
  ),
};

function FullSidebar({ open }: { open: boolean }) {
  return (
    <SidebarProvider defaultOpen={open}>
      <AppSidebar />
      <SidebarInset className="gap-4 p-4">
        <div className="flex items-center gap-4">
          <div data-testid="inset-sidebar-trigger">
            <SidebarTrigger />
          </div>
          <div data-testid="inset-app-sidebar-trigger">
            <AppSidebarToggleProvider>
              <AppSidebarTrigger />
            </AppSidebarToggleProvider>
          </div>
          <SidebarStateReadout />
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}

export const FullSidebarExpanded: Story = {
  render: () => <FullSidebar open />,
};

export const FullSidebarCollapsed: Story = {
  render: () => <FullSidebar open={false} />,
};
