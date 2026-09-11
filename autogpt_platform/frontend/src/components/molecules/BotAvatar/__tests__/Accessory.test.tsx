import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Accessory } from "../components/Accessory";
import { SHAPES, type AccessoryId } from "../helpers";
import { FRONT_POSE } from "../projection";

describe("avatar accessories", () => {
  it.each(["glasses", "headband", "ears"] as AccessoryId[])(
    "keeps %s geometry mounted while the pose changes",
    (accessory) => {
      const props = {
        accessory,
        anchors: SHAPES[0].anchors,
        deep: "#123456",
        outline: true,
        layer: "front" as const,
      };
      const { container, rerender } = render(
        <svg>
          <Accessory {...props} pose={FRONT_POSE} />
        </svg>,
      );
      const nodes = Array.from(container.querySelectorAll("path,circle"));
      expect(nodes.length).toBeGreaterThan(0);
      rerender(
        <svg>
          <Accessory {...props} pose={{ ...FRONT_POSE, yaw: 0.01 }} />
        </svg>,
      );
      const updated = Array.from(container.querySelectorAll("path,circle"));
      expect(updated).toHaveLength(nodes.length);
      for (let i = 0; i < nodes.length; i++) expect(updated[i]).toBe(nodes[i]);
    },
  );
});
