import { areaTint, seatTint, washTint } from "../areaColors";

// The values the approved mock uses.
describe("area colours", () => {
  it("tints an area's card 7% in light and 12% in dark", () => {
    expect(areaTint("sales", "light")).toBe("#FBF9F4");
    expect(areaTint("sales", "dark")).toBe("#332F2B");
    expect(areaTint("marketing", "light")).toBe("#FBF6F4");
  });

  it("fills an open seat with 8% of the area", () => {
    expect(seatTint("sales", "light")).toBe("#FBF8F2");
    expect(seatTint("sales", "dark")).toBe("#2D2A29");
  });

  it("starts the profile's wash at 22% of the area over the page grey", () => {
    expect(washTint("sales", "light")).toBe("#EAE1D2");
  });
});
