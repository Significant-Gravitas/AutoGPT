import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, test } from "vitest";
import { AutoGPTBubble } from "./AutoGPTBubble";

describe("AutoGPTBubble", () => {
  test("shows Autopilot as the sender label", () => {
    render(<AutoGPTBubble text="Hello there." animate={false} />);

    expect(screen.getByText("Autopilot")).toBeDefined();
    expect(screen.queryByText("AutoGPT")).toBeNull();
  });
});
