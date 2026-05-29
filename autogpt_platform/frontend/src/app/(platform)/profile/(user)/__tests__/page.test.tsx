import type { ReactNode } from "react";
import {
  render,
  screen,
  fireEvent,
  waitFor,
} from "@/tests/integrations/test-utils";
import {
  getGetV2GetUserProfileMockHandler,
  getPostV2UpdateUserProfileMockHandler,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import { server } from "@/mocks/mock-server";
import UserProfilePage from "../page";
import { beforeEach, describe, expect, test, vi } from "vitest";

const mockUseSupabase = vi.hoisted(() => vi.fn());

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: mockUseSupabase,
}));

const testUser = {
  id: "user-1",
  email: "user@example.com",
  app_metadata: {},
  user_metadata: {},
  aud: "authenticated",
  created_at: "2026-01-01T00:00:00.000Z",
};

describe("UserProfilePage", () => {
  beforeEach(() => {
    mockUseSupabase.mockReturnValue({
      user: testUser,
      isLoggedIn: true,
      isUserLoading: false,
      supabase: {},
    });
  });

  test("renders the existing profile and saves changes", async () => {
    let profile = {
      name: "Original Name",
      username: "original-user",
      description: "Original bio",
      links: ["https://example.com/1", "", "", "", ""],
      avatar_url: "",
      is_featured: false,
    };

    server.use(
      getGetV2GetUserProfileMockHandler(() => profile),
      getPostV2UpdateUserProfileMockHandler(async ({ request }) => {
        profile = (await request.json()) as typeof profile;
        return profile;
      }),
    );

    render(<UserProfilePage />);

    const displayName = await screen.findByLabelText("Display name");
    const handle = screen.getByLabelText("Handle");
    const bio = screen.getByLabelText("Bio");

    expect((displayName as HTMLInputElement).value).toBe("Original Name");
    expect((handle as HTMLInputElement).value).toBe("original-user");

    fireEvent.change(displayName, { target: { value: "Updated Name" } });
    fireEvent.change(handle, { target: { value: "updated-user" } });
    fireEvent.change(bio, { target: { value: "Updated bio" } });
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }));

    await waitFor(() => {
      expect(profile.name).toBe("Updated Name");
      expect(profile.username).toBe("updated-user");
      expect(profile.description).toBe("Updated bio");
    });
  });
});
