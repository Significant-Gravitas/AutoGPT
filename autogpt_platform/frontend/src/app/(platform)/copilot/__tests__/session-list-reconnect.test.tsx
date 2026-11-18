import { RecentChats } from "@/components/layout/AppSidebar/components/RecentChats/RecentChats";
import {
  type SidebarChat,
  sidebarBackend,
} from "@/components/layout/AppSidebar/components/RecentChats/__tests__/sidebarBackend";
import { SidebarProvider } from "@/components/molecules/Sidebar/Sidebar";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { server } from "@/mocks/mock-server";
import {
  AFTER_RECONNECT,
  installLiveSocket,
} from "@/tests/integrations/live-socket";
import { render, screen } from "@/tests/integrations/test-utils";
import { useEffect } from "react";
import { afterEach, expect, test, vi } from "vitest";
import { useCopilotNotifications } from "../useCopilotNotifications";

vi.mock("@/lib/auth/actions", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/auth/actions")>()),
  getWebSocketToken: async () => ({ token: "test-token" }),
}));

afterEach(() => {
  vi.unstubAllGlobals();
});

function CopilotWithSidebar() {
  const api = useBackendAPI();
  useCopilotNotifications(null);
  // The platform layout keeps the socket open (onboarding, wallet).
  useEffect(() => {
    void api.connectWebSocket();
  }, [api]);
  return (
    <SidebarProvider>
      <RecentChats />
    </SidebarProvider>
  );
}

test("a chat that finished while the live connection was down updates in the sidebar once it reconnects", async () => {
  let chats: SidebarChat[] = [
    { id: "chat-1", title: "New chat", is_processing: true },
  ];
  server.use(...sidebarBackend(() => chats));
  const live = installLiveSocket();

  render(<CopilotWithSidebar />);
  const first = await live.acceptNext();
  expect(await screen.findByText("New chat")).toBeDefined();

  // Its completion notification went out while the socket was down.
  chats = [{ id: "chat-1", title: "Plan my week", is_processing: false }];
  await live.drop(first);
  await live.acceptNext();

  expect(
    await screen.findByText("Plan my week", undefined, AFTER_RECONNECT),
  ).toBeDefined();
});
