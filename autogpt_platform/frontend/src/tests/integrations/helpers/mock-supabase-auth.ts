import type { User } from "@supabase/supabase-js";
import { useSupabaseStore } from "@/lib/supabase/hooks/useSupabaseStore";

export const mockUser: User = {
  id: "test-user-id",
  email: "test@example.com",
  aud: "authenticated",
  role: "authenticated",
  created_at: new Date().toISOString(),
  app_metadata: {},
  user_metadata: {},
};

export function mockAuthenticatedUser(user: Partial<User> = {}): User {
  const mergedUser = { ...mockUser, ...user };

  useSupabaseStore.setState({
    user: mergedUser,
    isUserLoading: false,
    hasLoadedUser: true,
    isValidating: false,
  });

  return mergedUser;
}

export function mockUnauthenticatedUser(): void {
  useSupabaseStore.setState({
    user: null,
    isUserLoading: false,
    hasLoadedUser: true,
    isValidating: false,
  });
}

export function resetAuthState(): void {
  useSupabaseStore.setState({
    user: null,
    isUserLoading: true,
    hasLoadedUser: false,
    isValidating: false,
  });
}
