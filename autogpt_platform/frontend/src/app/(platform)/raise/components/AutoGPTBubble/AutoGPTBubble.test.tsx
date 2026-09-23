import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, test } from "vitest";
import { AutoGPTBubble } from "./AutoGPTBubble";

describe("AutoGPTBubble", () => {
  test("shows Otto as the sender label", () => {
    render(<AutoGPTBubble text="Hello there." animate={false} />);

    expect(screen.getByText("Otto")).toBeDefined();
    expect(screen.queryByText("AutoGPT")).toBeNull();
  });
});
