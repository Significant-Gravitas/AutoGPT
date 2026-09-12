import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Accessory } from "../components/Accessory";
import { ACCESSORIES, SHAPES, type AccessoryId } from "../helpers";
import { FRONT_POSE } from "../projection";

const DRAWN = ACCESSORIES.filter((option) => option.id !== "none").map(
  (option) => option.id,
);
const LAYERS = ["front", "back"] as const;
const POSES = [
  FRONT_POSE,
  { yaw: 0.6, pitch: 0.2, roll: 0.1, bob: 1 },
  { yaw: -1.1, pitch: -0.3, roll: -0.1, bob: 0 },
  { yaw: 2.6, pitch: 0.35, roll: 0, bob: 2 },
];

function renderAccessory(
  accessory: AccessoryId,
  layer: (typeof LAYERS)[number],
  shapeIndex = 0,
) {
  return render(
    <svg>
      <Accessory
        accessory={accessory}
        anchors={SHAPES[shapeIndex].anchors}
        pose={FRONT_POSE}
        deep="#123456"
        outline={false}
        layer={layer}
      />
    </svg>,
  );
}

describe("avatar accessories", () => {
  it("draws every registered accessory", () => {
    expect(DRAWN.length).toBeGreaterThanOrEqual(20);
  });

  it.each(DRAWN)("renders %s in both layers with a stable testid", (id) => {
    for (const layer of LAYERS) {
      const { container, unmount } = renderAccessory(id, layer);
      const root = container.querySelector(`[data-testid="accessory-${id}"]`);
      expect(root).not.toBeNull();
      expect(root?.getAttribute("data-layer")).toBe(layer);
      unmount();
    }
  });

  it.each(DRAWN)("draws %s geometry in at least one layer", (id) => {
    const drawn = LAYERS.map((layer) => {
      const { container, unmount } = renderAccessory(id, layer);
      const count = container.querySelectorAll(
        "path,circle,ellipse,rect",
      ).length;
      unmount();
      return count;
    });
    expect(drawn[0] + drawn[1]).toBeGreaterThan(0);
  });

  it.each(DRAWN)("survives every pose and shape for %s", (id) => {
    for (const shapeIndex of SHAPES.keys()) {
      for (const pose of POSES) {
        for (const layer of LAYERS) {
          const { container, unmount } = render(
            <svg>
              <Accessory
                accessory={id}
                anchors={SHAPES[shapeIndex].anchors}
                pose={pose}
                deep="#123456"
                outline={shapeIndex % 2 === 0}
                layer={layer}
              />
            </svg>,
          );
          for (const node of container.querySelectorAll("path")) {
            expect(node.getAttribute("d") ?? "").not.toMatch(/NaN|Infinity/);
          }
          unmount();
        }
      }
    }
  });

  it("renders nothing for the none accessory", () => {
    const { container } = renderAccessory("none", "front");
    expect(container.querySelectorAll("path,circle,ellipse,rect")).toHaveLength(
      0,
    );
  });

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
