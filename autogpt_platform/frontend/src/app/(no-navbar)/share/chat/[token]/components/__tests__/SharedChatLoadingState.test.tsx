import { afterEach, describe, expect, test } from "vitest";

import { cleanup, render, screen } from "@/tests/integrations/test-utils";
import { SharedChatLoadingState } from "../SharedChatLoadingState";

afterEach(() => {
  cleanup();
});

describe("SharedChatLoadingState", () => {
  test("renders the loading copy", () => {
    render(<SharedChatLoadingState />);
    expect(screen.getByText(/loading shared chat/i)).toBeDefined();
  });
});
