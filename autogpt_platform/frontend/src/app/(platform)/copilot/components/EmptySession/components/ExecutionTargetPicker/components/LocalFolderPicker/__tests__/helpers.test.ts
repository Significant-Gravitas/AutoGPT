import type { ExecutorMachine } from "@/app/api/__generated__/models/executorMachine";
import { describe, expect, test } from "vitest";
import { LAPTOP, WORKSTATION } from "../../../__tests__/computers";
import { computerMenu } from "../helpers";

describe("the Computer menu's labels", () => {
  function labels(machines: ExecutorMachine[]) {
    return computerMenu(machines).options.map((option) => option.label);
  }

  test("each computer that shares a name shows the start of its own ID", () => {
    const twin = { ...WORKSTATION, machine_id: LAPTOP.machine_id };

    expect(labels([WORKSTATION, twin, LAPTOP])).toEqual([
      "Workstation · Windows · ID 1c49dced",
      "Workstation · Windows · ID 36d78179",
      "Laptop · macOS",
    ]);
  });

  test("IDs that start alike show as much as tells them apart", () => {
    const first = { ...WORKSTATION, machine_id: "office-pc-1c49dced" };
    const second = { ...WORKSTATION, machine_id: "office-pc-36d78179" };

    expect(labels([first, second])).toEqual([
      "Workstation · Windows · ID office-pc-1",
      "Workstation · Windows · ID office-pc-3",
    ]);
  });

  test("an ID that another one begins with still gets its own label", () => {
    const short = { ...WORKSTATION, machine_id: "office-pc" };
    const long = { ...WORKSTATION, machine_id: "office-pc-2" };

    expect(labels([short, long])).toEqual([
      "Workstation · Windows · ID office-pc",
      "Workstation · Windows · ID office-pc-",
    ]);
  });

  test("a name counts as the same name however its letters are encoded", () => {
    const composed = { ...WORKSTATION, display_name: "Caf\u00e9" };
    const decomposed = {
      ...WORKSTATION,
      display_name: "Cafe\u0301",
      machine_id: LAPTOP.machine_id,
    };

    expect(labels([composed, decomposed])).toEqual([
      "Caf\u00e9 · Windows · ID 1c49dced",
      "Caf\u00e9 · Windows · ID 36d78179",
    ]);
  });

  test("names that differ only in their spaces count as the same name", () => {
    const spaced = {
      ...WORKSTATION,
      display_name: " Office  PC ",
      machine_id: LAPTOP.machine_id,
    };

    expect(
      labels([{ ...WORKSTATION, display_name: "Office PC" }, spaced]),
    ).toEqual([
      "Office PC · Windows · ID 1c49dced",
      "Office PC · Windows · ID 36d78179",
    ]);
  });
});
