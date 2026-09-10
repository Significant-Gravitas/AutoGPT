import type { ComponentType } from "react";
import type { AccessoryId } from "../../helpers";
import { Antenna } from "./Antenna";
import { Badge } from "./Badge";
import { Bandana } from "./Bandana";
import { Beanie } from "./Beanie";
import { Bow } from "./Bow";
import { BowTie } from "./BowTie";
import { Cap } from "./Cap";
import { Crown } from "./Crown";
import { Earrings } from "./Earrings";
import { Ears } from "./Ears";
import { Flower } from "./Flower";
import { Glasses } from "./Glasses";
import { Halo } from "./Halo";
import { Headband } from "./Headband";
import { Headphones } from "./Headphones";
import { Headset } from "./Headset";
import { Monocle } from "./Monocle";
import { PartyHat } from "./PartyHat";
import { Propeller } from "./Propeller";
import { RoundGlasses } from "./RoundGlasses";
import { Star } from "./Star";
import { Sunglasses } from "./Sunglasses";
import { TopHat } from "./TopHat";
import type { AccessoryProps } from "./types";

export const ACCESSORY_COMPONENTS: Record<
  Exclude<AccessoryId, "none">,
  ComponentType<AccessoryProps>
> = {
  glasses: Glasses,
  roundglasses: RoundGlasses,
  sunglasses: Sunglasses,
  monocle: Monocle,
  headset: Headset,
  headphones: Headphones,
  crown: Crown,
  halo: Halo,
  cap: Cap,
  beanie: Beanie,
  tophat: TopHat,
  partyhat: PartyHat,
  bandana: Bandana,
  headband: Headband,
  ears: Ears,
  antenna: Antenna,
  propeller: Propeller,
  flower: Flower,
  bow: Bow,
  bowtie: BowTie,
  earrings: Earrings,
  badge: Badge,
  star: Star,
};
