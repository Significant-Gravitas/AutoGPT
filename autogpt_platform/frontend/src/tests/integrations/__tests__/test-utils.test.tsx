import { useQueryClient } from "@tanstack/react-query";
import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, test } from "vitest";

describe("test-utils render", () => {
  test("a rerender keeps the same query client, so its cache survives", () => {
    const seen: unknown[] = [];
    function CapturesClient() {
      seen.push(useQueryClient());
      return <div>hi</div>;
    }

    const view = render(<CapturesClient />);
    expect(screen.getByText("hi")).toBeDefined();
    view.rerender(<CapturesClient />);
    expect(seen).toHaveLength(2);
    expect(seen[0]).toBe(seen[1]);
  });
});
