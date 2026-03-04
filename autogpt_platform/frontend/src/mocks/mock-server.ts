import { setupServer } from "msw/node";
import { mockHandlers } from "./mock-handlers";

export const server = setupServer(...mockHandlers);
