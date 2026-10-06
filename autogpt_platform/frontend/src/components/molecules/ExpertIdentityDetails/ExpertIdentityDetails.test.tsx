import { render, screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { ExpertIdentityDetails } from "./ExpertIdentityDetails";

describe("ExpertIdentityDetails", () => {
  test.each(["card", "page"] as const)(
    "falls back to the role label on the name line in %s size",
    (size) => {
      render(
        <ExpertIdentityDetails
          name="Jules"
          role="Social & Content Repurposing"
          size={size}
        />,
      );

      expect(screen.getAllByText(/Social Media Manager/)).toHaveLength(1);
      expect(screen.queryByText("AI Expert")).toBeNull();
    },
  );

  test.each(["card", "page"] as const)(
    "puts the job title on the name line in %s size, in place of the kind",
    (size) => {
      render(
        <ExpertIdentityDetails
          name="Jules"
          role="Social & Content Repurposing"
          jobTitle="Social Media Manager"
          size={size}
        />,
      );

      const title = screen.getByText(/Social Media Manager/);
      expect(title.closest("div")?.textContent).toBe(
        "Jules\u2022 Social Media Manager",
      );
      expect(screen.queryByText("AI Expert")).toBeNull();
      expect(screen.queryByText("Social media")).toBeNull();
    },
  );

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
      screen.getByText("Social Media Manager").classList.contains("leading-4"),
    ).toBe(true);
    expect(container.querySelector("svg")).toBeNull();
  });
});

test("does not grant Otto's identity exception from a specialist title", () => {
  render(<ExpertIdentityDetails name="My Expert" role="Head of AI" />);
  expect(screen.getByText(/Head of AI/)).toBeDefined();
  expect(screen.queryByText("Your personal Head of AI")).toBeNull();
});

test.each(["compact", "card", "page"] as const)(
  "discloses AI for an Expert without a role or title in %s size",
  (size) => {
    render(
      <ExpertIdentityDetails
        name="My Expert"
        role={null}
        jobTitle={null}
        size={size}
      />,
    );
    expect(screen.getByText("AI Expert")).toBeDefined();
    expect(screen.queryByText("Your personal Head of AI")).toBeNull();
  },
);

test("keeps Otto's disclosure without a role or title", () => {
  render(<ExpertIdentityDetails name="Otto" isOtto />);
  expect(screen.getByText("Your personal Head of AI")).toBeDefined();
  expect(screen.queryByText("AI Expert")).toBeNull();
});

test("shows only name and role for a compact specialist", () => {
  render(
    <ExpertIdentityDetails name="Mina" jobTitle="Bookkeeper" size="compact" />,
  );
  expect(screen.getByText("Mina")).toBeDefined();
  expect(screen.getByText("Bookkeeper")).toBeDefined();
  expect(screen.queryByText("AI Expert")).toBeNull();
});

test.each(["compact", "card", "page"] as const)(
  "avoids repeating Otto's role in %s size",
  (size) => {
    render(
      <ExpertIdentityDetails
        name="Otto"
        isOtto
        role="Head of AI"
        size={size}
      />,
    );
    expect(screen.getByText(/Head of AI/)).toBeDefined();
    expect(screen.queryByText("Your personal Head of AI")).toBeNull();
  },
);
