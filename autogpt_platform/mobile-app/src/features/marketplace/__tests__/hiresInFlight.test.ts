import { claimHire, releaseHire } from "../hiresInFlight";

describe("claimHire", () => {
  afterEach(() => {
    releaseHire("t1");
    releaseHire("t2");
  });

  it("refuses a second claim for a hire already running", () => {
    expect(claimHire("t1")).toBe(true);
    expect(claimHire("t1")).toBe(false);
  });

  it("lets different experts be hired at once", () => {
    expect(claimHire("t1")).toBe(true);
    expect(claimHire("t2")).toBe(true);
  });

  it("allows a new hire once the last one is released", () => {
    expect(claimHire("t1")).toBe(true);
    releaseHire("t1");
    expect(claimHire("t1")).toBe(true);
  });
});
