import type { AccessoryId, ShapeAnchors } from "../helpers";
import type { Layer } from "../parts";
import { ellipsoidFor, type Pose } from "../projection";
import { ACCESSORY_COMPONENTS } from "./accessories/registry";

interface Props {
  accessory: AccessoryId;
  anchors: ShapeAnchors;
  pose: Pose;
  deep: string;
  outline: boolean;
  layer: Layer;
}

// Each accessory draws itself twice — once behind the head and once in front
// — and decides per part which half it belongs to for the current pose.
export function Accessory({
  accessory,
  anchors,
  pose,
  deep,
  outline,
  layer,
}: Props) {
  if (accessory === "none") return null;
  const Part = ACCESSORY_COMPONENTS[accessory];
  if (!Part) return null;

  return (
    <g data-testid={`accessory-${accessory}`} data-layer={layer}>
      <Part
        anchors={anchors}
        body={ellipsoidFor(anchors)}
        pose={pose}
        deep={deep}
        outline={outline}
        layer={layer}
      />
    </g>
  );
}
