import { getListExpertIdentitiesMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { SidebarProvider } from "@/components/molecules/Sidebar/Sidebar";
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
const identitiesRequest = vi.fn(() => [
  {
    id: "expert-maria",
    name: "Maria Ruiz",
    avatar_url: null,
    role: "Marketing",
    job_title: "Growth Lead",
    is_archived: false,
  },
]);

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

function expectNoExperts() {
  expect(screen.queryByText("Experts")).toBeNull();
  expect(screen.queryByText("Maria Ruiz")).toBeNull();
}

beforeEach(() => {
  vi.clearAllMocks();
  useAuthStore.setState(useAuthStore.getInitialState());
  getQueryClient().clear();
  vi.mocked(getCurrentUser).mockResolvedValue({ user: null });
  vi.mocked(validateSession).mockResolvedValue({ user: null, isValid: false });
  vi.mocked(serverLogout).mockResolvedValue({ success: true });
  server.use(getListExpertIdentitiesMockHandler(identitiesRequest));
});

afterEach(() => {
  useAuthStore.getState().cleanup();
  getQueryClient().clear();
});

describe("AppSidebar auth state", () => {
  it("does not fetch or show experts while auth loads or after finding no user", async () => {
    const auth = deferred<Awaited<ReturnType<typeof getCurrentUser>>>();
    vi.mocked(getCurrentUser).mockReturnValue(auth.promise);
    renderSidebar();

    expect(useAuthStore.getState().isUserLoading).toBe(true);
    expectNoExperts();
    expect(identitiesRequest).not.toHaveBeenCalled();

    await act(async () => {
      auth.resolve({ user: null });
      await useAuthStore.getState().initializationPromise;
    });

    expect(useAuthStore.getState().isUserLoading).toBe(false);
    expectNoExperts();
    expect(identitiesRequest).not.toHaveBeenCalled();
  });

  it("links to Tasks only for a signed-in user", async () => {
    const auth = deferred<Awaited<ReturnType<typeof getCurrentUser>>>();
    vi.mocked(getCurrentUser).mockReturnValue(auth.promise);
    renderSidebar();
    expect(screen.queryByRole("link", { name: "Tasks" })).toBeNull();

    await act(async () => {
      auth.resolve({ user });
      await useAuthStore.getState().initializationPromise;
    });

    const tasks = await screen.findByRole("link", { name: "Tasks" });
    expect(tasks.getAttribute("href")).toBe("/tasks");
  });

  it("fetches the expert roster only after the server returns an authenticated user", async () => {
    const auth = deferred<Awaited<ReturnType<typeof getCurrentUser>>>();
    vi.mocked(getCurrentUser).mockReturnValue(auth.promise);
    renderSidebar();
    expect(identitiesRequest).not.toHaveBeenCalled();

    await act(async () => {
      auth.resolve({ user });
      await useAuthStore.getState().initializationPromise;
    });

    expect(useAuthStore.getState().isUserLoading).toBe(false);
    expect(await screen.findByText("Maria Ruiz")).toBeDefined();
    expect(screen.getByText("Experts")).toBeDefined();
    expect(identitiesRequest).toHaveBeenCalledOnce();
  });

  it("removes experts when background validation rejects the session", async () => {
    vi.mocked(getCurrentUser).mockResolvedValue({ user });
    const validation = deferred<Awaited<ReturnType<typeof validateSession>>>();
    vi.mocked(validateSession).mockReturnValue(validation.promise);
    renderSidebar();
    await screen.findByText("Maria Ruiz");

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

    expectNoExperts();
    expect(identitiesRequest).toHaveBeenCalledOnce();
  });

  it("keeps experts until the server confirms the sign-out, then removes them", async () => {
    vi.mocked(getCurrentUser).mockResolvedValue({ user });
    const logout = deferred<Awaited<ReturnType<typeof serverLogout>>>();
    vi.mocked(serverLogout).mockReturnValue(logout.promise);
    renderSidebar();
    await screen.findByText("Maria Ruiz");

    let loggingOut!: Promise<void>;
    act(() => {
      loggingOut = useAuthStore.getState().logOut();
    });
    // Until the server has the request, the session may still be live.
    await waitFor(() => expect(serverLogout).toHaveBeenCalled());
    expect(useAuthStore.getState().user).toBe(user);

    await act(async () => {
      logout.resolve({ success: true });
      await loggingOut;
    });
    expect(useAuthStore.getState().user).toBeNull();
    expectNoExperts();
    expect(identitiesRequest).toHaveBeenCalledOnce();
  });

  it("removes experts when another tab logs out", async () => {
    vi.mocked(getCurrentUser).mockResolvedValue({ user });
    renderSidebar();
    await screen.findByText("Maria Ruiz");

    act(() => {
      window.dispatchEvent(
        new StorageEvent("storage", { key: Key.LOGOUT, newValue: "1" }),
      );
    });

    expect(useAuthStore.getState().user).toBeNull();
    expectNoExperts();
    expect(identitiesRequest).toHaveBeenCalledOnce();
  });
});
