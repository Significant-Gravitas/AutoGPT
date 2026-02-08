import { beforeAll, afterAll, afterEach } from "vitest";
import { server } from "@/mocks/mock-server";
import { mockNextjsModules } from "./setup-nextjs-mocks";
import { mockSupabaseRequest } from "./mock-supabase-request";

beforeAll(() => {
  mockNextjsModules();
  mockSupabaseRequest(); // If you need user's data - please mock supabase actions in your specific test - it sends null user [It's only to avoid cookies() call]
  return server.listen({ onUnhandledRequest: "error" });
});
afterEach(() => server.resetHandlers());
afterAll(() => server.close());
