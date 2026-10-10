import { describe, expect, test, vi } from "vitest";

import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import type {
  BlockIOObjectSubSchema,
  BlockIOSimpleTypeSubSchema,
  BlockIOStringSubSchema,
  BlockIOSubSchema,
} from "@/lib/autogpt-server-api/types";
import { RunAgentInputs } from "../RunAgentInputs";

const channels: BlockIOObjectSubSchema = {
  type: "object",
  properties: {
    email: { type: "boolean", title: "Email" },
    sms: { type: "boolean", title: "SMS" },
  },
};

const tone: BlockIOStringSubSchema = {
  type: "string",
  enum: ["formal", "casual"],
};

function optional(schema: BlockIOSimpleTypeSubSchema): BlockIOSubSchema {
  return { anyOf: [schema, { type: "null" }] };
}

describe("RunAgentInputs", () => {
  test.each([
    ["required", channels],
    ["optional", optional(channels)],
  ])(
    "%s yes/no group renders its toggles and reports every key",
    (_, schema) => {
      const onChange = vi.fn();
      render(
        <RunAgentInputs
          schema={{ ...schema, title: "Channels" }}
          value={null}
          onChange={onChange}
        />,
      );

      screen.getByRole("checkbox", { name: "SMS" });
      fireEvent.click(screen.getByRole("checkbox", { name: "Email" }));

      expect(onChange).toHaveBeenCalledWith({ email: true, sms: false });
    },
  );

  test.each([
    ["required", tone],
    ["optional", optional(tone)],
  ])("%s enum renders a select of its values", async (_, schema) => {
    const onChange = vi.fn();
    render(
      <RunAgentInputs
        schema={{ ...schema, title: "Tone" }}
        value={null}
        onChange={onChange}
      />,
    );

    fireEvent.click(screen.getByRole("combobox"));
    fireEvent.click(await screen.findByRole("option", { name: "casual" }));

    expect(onChange).toHaveBeenCalledWith("casual");
  });
});
