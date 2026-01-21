import { beforeAll, afterAll, afterEach } from "vitest";
import { server } from "@/mocks/mock-server";

beforeAll(() => server.listen({ onUnhandledRequest: "bypass" }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());
