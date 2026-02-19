import { beforeAll, afterAll, afterEach } from "vitest";
import { server } from "@/mocks/mock-server";
import { mockNextjsModules } from "./setup-nextjs-mocks";
import { mockSupabaseRequest } from "./mock-supabase-request";
import "@testing-library/jest-dom";
import { suppressReactQueryUpdateWarning } from "./helpers/suppress-react-query-update-warning";

let restoreConsoleError: (() => void) | null = null;

beforeAll(() => {
  mockNextjsModules();
  mockSupabaseRequest();
  restoreConsoleError = suppressReactQueryUpdateWarning();
  server.listen({ onUnhandledRequest: "error" });
});

afterEach(() => {
  server.resetHandlers();
});

afterAll(() => {
  restoreConsoleError?.();
  server.close();
});
