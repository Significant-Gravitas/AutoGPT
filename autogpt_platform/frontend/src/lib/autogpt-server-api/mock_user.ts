import { User } from "@supabase/supabase-js";

// Create a mock Supabase user for testing and storybook
export const mockUser: User = {
  id: "mock-user-id-123",
  app_metadata: { provider: "email" },
  user_metadata: { name: "Mock User" },
  aud: "authenticated",
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  email: "mock@example.com",
  role: "admin",
};
