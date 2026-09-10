import type { ShapeAnchors } from "../../helpers";
import type { Layer } from "../../parts";
import type { Ellipsoid, Pose } from "../../projection";

export interface AccessoryProps {
  anchors: ShapeAnchors;
  body: Ellipsoid;
  pose: Pose;
  deep: string;
  outline: boolean;
  layer: Layer;
}
