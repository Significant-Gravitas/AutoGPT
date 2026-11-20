import type { Expert } from "../../../lib/api/experts";
import {
  claimHire,
  hireStateFor,
  readHireState,
  recordHire,
  releaseHire,
  hiresSince,
} from "../hireStore";

// The store is module state; each test starts from a sign-in no other test uses.
let generation = 100;

beforeEach(() => {
  generation += 1;
});

const expert = { id: "e1", name: "Alex" } as Expert;

describe("claimHire", () => {
  it("refuses a second claim for a hire already running", () => {
    expect(claimHire("t1", generation)).toBe(true);
    expect(claimHire("t1", generation)).toBe(false);
  });

  it("lets different experts be hired at once", () => {
    expect(claimHire("t1", generation)).toBe(true);
    expect(claimHire("t2", generation)).toBe(true);
  });

  it("allows a new hire once the last one is released", () => {
    expect(claimHire("t1", generation)).toBe(true);
    releaseHire("t1", generation);
    expect(claimHire("t1", generation)).toBe(true);
  });

  it("starts clean for the next sign-in, and the last one's release can't touch it", () => {
    expect(claimHire("t1", generation)).toBe(true);
    const next = generation + 1;
    expect(claimHire("t1", next)).toBe(true);
    releaseHire("t1", generation);
    expect(claimHire("t1", next)).toBe(false);
    generation = next;
  });
});

describe("recordHire", () => {
  it("keeps a confirmed hire for this sign-in only", () => {
    claimHire("t1", generation);
    recordHire("t1", expert, generation);
    const snapshot = hireStateFor(readHireState(), generation);
    expect(snapshot.hired.get("t1")).toBe(expert);
    expect(hireStateFor(readHireState(), generation + 1).hired.size).toBe(0);
  });
});

describe("hiresSince", () => {
  it("keeps a hire until the team list loads after it, then leaves it to the list", () => {
    claimHire("t1", generation);
    recordHire("t1", expert, generation);
    const current = hireStateFor(readHireState(), generation);
    const at = current.hiredAt.get("t1") ?? 0;
    expect([...hiresSince(current, at - 1).keys()]).toEqual(["t1"]);
    expect(hiresSince(current, at + 1).size).toBe(0);
    expect(current.hired.get("t1")).toBe(expert);
  });
});
