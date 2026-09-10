import { describe, it, expect } from "vitest";
import {
  connectionHandleOffset,
  CONNECTION_STEP_OFFSET,
  createConnectionSteps,
} from "./connections";

describe("connectionHandleOffset", () => {
  it("shifts left placements further left along x", () => {
    expect(
      connectionHandleOffset({ placement: "left", x: 100, y: 50 }),
    ).toEqual({ x: 100 - CONNECTION_STEP_OFFSET });
  });

  it("shifts right placements further right along x", () => {
    expect(
      connectionHandleOffset({ placement: "right", x: 100, y: 50 }),
    ).toEqual({ x: 100 + CONNECTION_STEP_OFFSET });
  });

  it("shifts top placements further up along y", () => {
    expect(connectionHandleOffset({ placement: "top", x: 100, y: 50 })).toEqual(
      {
        y: 50 - CONNECTION_STEP_OFFSET,
      },
    );
  });

  it("shifts bottom placements further down along y", () => {
    expect(
      connectionHandleOffset({ placement: "bottom", x: 100, y: 50 }),
    ).toEqual({ y: 50 + CONNECTION_STEP_OFFSET });
  });

  it("respects the base side of aligned placements like left-start", () => {
    expect(
      connectionHandleOffset({ placement: "left-start", x: 100, y: 50 }),
    ).toEqual({ x: 100 - CONNECTION_STEP_OFFSET });
  });

  it("returns no offset for unknown placements", () => {
    expect(
      connectionHandleOffset({ placement: "center", x: 100, y: 50 }),
    ).toEqual({});
  });
});

describe("createConnectionSteps floating offset", () => {
  const tour = { next: () => {}, back: () => {}, show: () => {} };

  it("attaches the connection offset middleware to the connect steps", () => {
    const steps = createConnectionSteps(tour) as Array<{
      id?: string;
      floatingUIOptions?: {
        middleware?: Array<{
          name: string;
          fn: (state: { placement: string; x: number; y: number }) => {
            x?: number;
            y?: number;
          };
        }>;
      };
    }>;

    const output = steps.find((s) => s.id === "connect-blocks-output");
    const input = steps.find((s) => s.id === "connect-blocks-input");

    const middleware = output?.floatingUIOptions?.middleware?.[0];
    expect(middleware?.name).toBe("connectionHandleOffset");
    expect(input?.floatingUIOptions?.middleware?.[0]?.name).toBe(
      "connectionHandleOffset",
    );

    // Invoking the middleware fn exercises the wiring to connectionHandleOffset.
    expect(middleware?.fn({ placement: "left", x: 200, y: 30 })).toEqual({
      x: 200 - CONNECTION_STEP_OFFSET,
    });
  });
});
