import { render } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { SubmissionsLoading } from "./SubmissionsLoading";

describe("SubmissionsLoading", () => {
  test("shows three placeholder rows while submissions load", () => {
    const { container } = render(<SubmissionsLoading />);
    const root = container.firstElementChild;

    expect(root?.children).toHaveLength(3);
  });
});
