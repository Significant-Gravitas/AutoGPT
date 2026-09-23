import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, test } from "vitest";
import { SoulPreviewPanel } from "./SoulPreviewPanel";

const baseProps = {
  role: null,
  avatarUrl: null,
  color: null,
  about: null,
  voiceLabel: null,
  kit: null,
};

describe("SoulPreviewPanel", () => {
  test("keeps a heading in the outline before a name is picked", () => {
    render(<SoulPreviewPanel {...baseProps} name="" />);

    expect(
      screen.getByRole("heading", { level: 2, name: "No name yet" }),
    ).toBeDefined();
  });

  test("keeps the same heading element once a name is picked", () => {
    render(<SoulPreviewPanel {...baseProps} name="Maria" />);

    expect(
      screen.getByRole("heading", { level: 2, name: "Maria" }),
    ).toBeDefined();
  });
});
