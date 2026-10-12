import { getGetV2ListSessionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { SidebarProvider } from "@/components/ui/sidebar";
import {
  getCurrentUser,
  serverLogout,
  validateSession,
} from "@/lib/auth/actions";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { User } from "@/lib/auth/types";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { server } from "@/mocks/mock-server";
import { Key } from "@/services/storage/local-storage";
import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppSidebar } from "../AppSidebar";

vi.mock("@/lib/auth/actions", () => ({
  getCurrentUser: vi.fn(),
  validateSession: vi.fn(),
  serverLogout: vi.fn(),
}));

vi.mock("../components/SidebarUserActions/SidebarUserActions", () => ({
  SidebarUserActions: () => null,
}));

const user: User = {
  id: "user-1",
  email: "alice@example.com",
  role: "user",
  user_metadata: {},
};
const sessionsRequest = vi.fn(() => ({
  sessions: [
    {
      id: "session-1",
      title: "My real conversation",
      is_processing: false,
      created_at: "2026-06-30T00:00:00Z",
      updated_at: "2026-06-30T00:00:00Z",
    },
  ],
  total: 1,
}));

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

function renderSidebar() {
  return render(
    <QueryClientProvider client={getQueryClient()}>
      <BackendAPIProvider>
        <NuqsTestingAdapter>
          <SidebarProvider>
            <AppSidebar />
          </SidebarProvider>
        </NuqsTestingAdapter>
      </BackendAPIProvider>
    </QueryClientProvider>,
  );
}

function expectNoRecentChats() {
  expect(screen.queryByText("Recent chats")).toBeNull();
  expect(screen.queryByText("My real conversation")).toBeNull();
  expect(screen.queryByText(/no conversations yet/i)).toBeNull();
}

beforeEach(() => {
  vi.clearAllMocks();
  useAuthStore.setState(useAuthStore.getInitialState());
  getQueryClient().clear();
  vi.mocked(getCurrentUser).mockResolvedValue({ user: null });
  vi.mocked(validateSession).mockResolvedValue({ user: null, isValid: false });
  vi.mocked(serverLogout).mockResolvedValue({ success: true });
  server.use(getGetV2ListSessionsMockHandler200(sessionsRequest));
});

afterEach(() => {
  useAuthStore.getState().cleanup();
  getQueryClient().clear();
});

describe("AppSidebar auth state", () => {
  it("does not fetch or show chats while auth loads or after finding no user", async () => {
    const auth = deferred<Awaited<ReturnType<typeof getCurrentUser>>>();
    vi.mocked(getCurrentUser).mockReturnValue(auth.promise);
    renderSidebar();

    expect(useAuthStore.getState().isUserLoading).toBe(true);
    expectNoRecentChats();
    expect(sessionsRequest).not.toHaveBeenCalled();

    await act(async () => {
      auth.resolve({ user: null });
      await useAuthStore.getState().initializationPromise;
    });

    expect(useAuthStore.getState().isUserLoading).toBe(false);
    expectNoRecentChats();
    expect(sessionsRequest).not.toHaveBeenCalled();
  });

  it("fetches chats only after the server returns an authenticated user", async () => {
    const auth = deferred<Awaited<ReturnType<typeof getCurrentUser>>>();
    vi.mocked(getCurrentUser).mockReturnValue(auth.promise);
    renderSidebar();
    expect(sessionsRequest).not.toHaveBeenCalled();

    await act(async () => {
      auth.resolve({ user });
      await useAuthStore.getState().initializationPromise;
    });

    expect(useAuthStore.getState().isUserLoading).toBe(false);
    expect(await screen.findByText("My real conversation")).toBeDefined();
    expect(screen.getByText("Recent chats")).toBeDefined();
    expect(sessionsRequest).toHaveBeenCalledOnce();
  });

  it("removes chats when background validation rejects the session", async () => {
    vi.mocked(getCurrentUser).mockResolvedValue({ user });
    const validation = deferred<Awaited<ReturnType<typeof validateSession>>>();
    vi.mocked(validateSession).mockReturnValue(validation.promise);
    renderSidebar();
    await screen.findByText("My real conversation");

    let validating!: Promise<boolean>;
    act(() => {
      validating = useAuthStore.getState().validateSession({ force: true });
    });
    expect(useAuthStore.getState()).toMatchObject({
      user,
      isUserLoading: false,
      isValidating: true,
    });
    await act(async () => {
      validation.resolve({ user: null, isValid: false });
      await validating;
    });

    expectNoRecentChats();
    expect(sessionsRequest).toHaveBeenCalledOnce();
  });

  it("removes chats before the server finishes logging out", async () => {
    vi.mocked(getCurrentUser).mockResolvedValue({ user });
    const logout = deferred<Awaited<ReturnType<typeof serverLogout>>>();
    vi.mocked(serverLogout).mockReturnValue(logout.promise);
    renderSidebar();
    await screen.findByText("My real conversation");

    let loggingOut!: Promise<void>;
    act(() => {
      loggingOut = useAuthStore.getState().logOut();
    });
    await waitFor(() => expect(useAuthStore.getState().user).toBeNull());
    expectNoRecentChats();
    await act(async () => {
      logout.resolve({ success: true });
      await loggingOut;
    });
    expectNoRecentChats();
    expect(sessionsRequest).toHaveBeenCalledOnce();
  });

  it("removes chats when another tab logs out", async () => {
    vi.mocked(getCurrentUser).mockResolvedValue({ user });
    renderSidebar();
    await screen.findByText("My real conversation");

    act(() => {
      window.dispatchEvent(
        new StorageEvent("storage", { key: Key.LOGOUT, newValue: "1" }),
      );
    });

    expect(useAuthStore.getState().user).toBeNull();
    expectNoRecentChats();
    expect(sessionsRequest).toHaveBeenCalledOnce();
  });
});
