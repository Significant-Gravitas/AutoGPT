import { render, screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { ExpertIdentityDetails } from "./ExpertIdentityDetails";

describe("ExpertIdentityDetails", () => {
  test.each(["card", "page"] as const)(
    "uses one neutral area chip in %s size",
    (size) => {
      render(
        <ExpertIdentityDetails
          name="Jules"
          role="Social & Content Repurposing"
          size={size}
        />,
      );

      const area = screen.getByText("Social media");
      const chip = area.parentElement;
      expect(screen.getAllByText("Social media")).toHaveLength(1);
      expect(chip?.classList.contains("bg-zinc-50")).toBe(true);
      expect(chip?.classList.contains("rounded-full")).toBe(true);
      expect(chip?.querySelector("svg")).not.toBeNull();
    },
  );

  test("shows the job title in the chip and keeps the area's icon", () => {
    render(
      <ExpertIdentityDetails
        name="Jules"
        role="Social & Content Repurposing"
        jobTitle="Social Media Manager"
      />,
    );

    const chip = screen.getByText("Social Media Manager").parentElement;
    expect(chip?.querySelector("svg")).not.toBeNull();
    expect(screen.queryByText("Social media")).toBeNull();
  });

  test("shows the job title as the compact text", () => {
    render(
      <ExpertIdentityDetails
        name="Jules"
        role="Social & Content Repurposing"
        jobTitle="Social Media Manager"
        size="compact"
      />,
    );

    expect(screen.getByText("Social Media Manager")).toBeDefined();
    expect(screen.queryByText("Social media")).toBeNull();
  });

  test("keeps the compact area as small plain text", () => {
    const { container } = render(
      <ExpertIdentityDetails
        name="Jules"
        role="Social & Content Repurposing"
        size="compact"
      />,
    );
    expect(
      screen.getByText("Social media").classList.contains("leading-4"),
    ).toBe(true);
    expect(container.querySelector("svg")).toBeNull();
  });
});
