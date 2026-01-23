import { beforeAll, afterAll, afterEach } from "vitest";
import { server } from "@/mocks/mock-server";
import { mockNextjsModules } from "./setup-nextjs-mocks";
import { mockSupabaseRequest } from "./mock-supabase-request";
import "@testing-library/jest-dom";
import { suppressReactQueryUpdateWarning } from "./helpers/supress-react-query-update-warning";

beforeAll(() => {
  mockNextjsModules();
  mockSupabaseRequest();
  const restoreConsoleError = suppressReactQueryUpdateWarning();
  afterAll(() => {
    restoreConsoleError();
  });
  return server.listen({ onUnhandledRequest: "error" });
});
afterEach(() => {server.resetHandlers()});
afterAll(() => server.close());
