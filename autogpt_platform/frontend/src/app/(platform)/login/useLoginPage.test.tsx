import { describe, it, expect, vi } from "vitest";
import { renderHook } from "@testing-library/react";
import { useLoginPage } from "./useLoginPage";

// Mock next/navigation
vi.mock("next/navigation", () => ({
  useRouter: vi.fn(() => ({
    replace: vi.fn(),
  })),
  useSearchParams: vi.fn(() => ({
    get: vi.fn(),
  })),
}));

// Mock login actions
vi.mock("./actions", () => ({
  login: vi.fn(),
}));

// Mock supabase hook
vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: vi.fn(() => ({
    user: null,
    isUserLoading: false,
    isLoggedIn: false,
    supabase: {},
  })),
}));

// Mock toast
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: vi.fn(() => ({
    toast: vi.fn(),
  })),
}));

// Mock environment
vi.mock("@/services/environment", () => ({
  environment: {
    isCloud: vi.fn(() => true),
    getBehaveAs: vi.fn(() => "prod"),
  },
}));

describe("useLoginPage - #11018 infinite loading fix", () => {
  it("should be defined", () => {
    const { result } = renderHook(() => useLoginPage());
    expect(result.current).toBeDefined();
    expect(result.current.handleSubmit).toBeDefined();
    expect(typeof result.current.isLoading).toBe("boolean");
  });
});
