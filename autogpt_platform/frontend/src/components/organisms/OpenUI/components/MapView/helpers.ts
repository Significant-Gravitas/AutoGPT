import { z } from "zod/v4";
import {
  mapLocationSchema,
  type MapLocation,
} from "@/lib/openui/catalog-sections";

export const indexedLocationsSchema = z
  .array(
    z.object({
      index: z.number().int().nonnegative(),
      place: mapLocationSchema,
    }),
  )
  .max(50);

export type IndexedLocation = z.infer<typeof indexedLocationsSchema>[number];

export function validLocations(locations: MapLocation[] = []) {
  return locations.slice(0, 50).flatMap((place, index) => {
    const result = mapLocationSchema.safeParse(place);
    return result.success ? [{ index, place: result.data }] : [];
  });
}

export function placeCoordinates(locations: IndexedLocation[]) {
  const longitudes = locations
    .map(({ place }) => place.longitude)
    .sort((a, b) => a - b);
  if (!longitudes.length) return [];
  let largestGap = -1;
  let start = longitudes[0];
  longitudes.forEach((longitude, index) => {
    const next = longitudes[(index + 1) % longitudes.length];
    const gap = next - longitude + (index === longitudes.length - 1 ? 360 : 0);
    if (gap > largestGap) {
      largestGap = gap;
      start = next;
    }
  });
  return locations.map(({ index, place }) => ({
    index,
    place,
    coordinates: [
      place.latitude,
      place.longitude < start ? place.longitude + 360 : place.longitude,
    ] as [number, number],
  }));
}
