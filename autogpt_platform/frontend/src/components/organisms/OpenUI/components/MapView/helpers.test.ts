import { describe, expect, it, vi } from "vitest";
import { placeCoordinates, validLocations } from "./helpers";
import { createPlaceMarker } from "./leafletHelpers";

const place = {
  name: "Place",
  latitude: 10,
  longitude: 179,
  detail: "",
  category: "",
};

describe("geographic map data", () => {
  it("fits points on either side of the antimeridian together", () => {
    const points = placeCoordinates(
      validLocations([place, { ...place, longitude: -179 }]),
    );
    expect(Math.abs(points[0].coordinates[1] - points[1].coordinates[1])).toBe(
      2,
    );
    expect(points[1].place.longitude).toBe(-179);
  });

  it("ignores incomplete streamed locations without renumbering valid markers", () => {
    const points = validLocations([{ ...place, latitude: 1000 }, place]);
    expect(points).toEqual([{ index: 1, place }]);
  });

  it("treats model-supplied names as text and returns the selected place index", () => {
    const onSelect = vi.fn();
    const name = '<img src=x onerror="alert(1)">';
    const [point] = placeCoordinates([{ index: 4, place: { ...place, name } }]);
    const marker = createPlaceMarker(point, onSelect);
    const content = marker.getTooltip()?.getContent();
    expect(content).toBeInstanceOf(HTMLElement);
    expect((content as HTMLElement).textContent).toBe(name);
    expect((content as HTMLElement).querySelector("img")).toBeNull();
    marker.fire("click");
    expect(onSelect).toHaveBeenCalledWith(4);
  });
});
