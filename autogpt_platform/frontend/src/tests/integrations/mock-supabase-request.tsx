import { vi } from "vitest";

const mockSupabaseClient = {
  auth: {
    getUser: vi.fn().mockResolvedValue({
      data: { user: null },
      error: null,
    }),
    getSession: vi.fn().mockResolvedValue({
      data: { session: null },
      error: null,
    }),
    signOut: vi.fn().mockResolvedValue({ error: null }),
    refreshSession: vi.fn().mockResolvedValue({
      data: { session: null, user: null },
      error: null,
    }),
  },
};

export const mockSupabaseRequest = () => {
  vi.mock("@/lib/supabase/server/getServerSupabase", () => ({
    getServerSupabase: vi.fn().mockResolvedValue(mockSupabaseClient),
  }));
};
