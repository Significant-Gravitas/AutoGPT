import { washTint } from "../areaColors";

// The values the approved mock uses.
describe("area colours", () => {
  it("starts the profile's wash at 12% of the area over the page grey", () => {
    expect(washTint("sales", "light")).toBe("#EEE9E2");
  });
});
