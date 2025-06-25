import { Skeleton } from "@/components/ui/skeleton";
import { render, screen } from "@testing-library/react";
import { userEvent } from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

describe("Skeleton Component", () => {
  it("renders skeleton element", () => {
    render(<Skeleton data-testid="skeleton" />);

    const skeleton = screen.getByTestId("skeleton");
    expect(skeleton).toBeInTheDocument();
  });

  it("applies custom className", () => {
    render(<Skeleton className="custom-skeleton" data-testid="skeleton" />);

    const skeleton = screen.getByTestId("skeleton");
    expect(skeleton).toHaveClass("custom-skeleton");
  });

  it("passes through HTML attributes", () => {
    render(
      <Skeleton
        data-testid="skeleton"
        role="progressbar"
        aria-label="Loading content"
      />,
    );

    const skeleton = screen.getByTestId("skeleton");
    expect(skeleton).toHaveAttribute("role", "progressbar");
    expect(skeleton).toHaveAttribute("aria-label", "Loading content");
  });

  it("renders as a div element", () => {
    render(<Skeleton data-testid="skeleton" />);

    const skeleton = screen.getByTestId("skeleton");
    expect(skeleton.tagName).toBe("DIV");
  });

  it("can contain children", () => {
    render(
      <Skeleton data-testid="skeleton">
        <span>Loading...</span>
      </Skeleton>,
    );

    expect(screen.getByText("Loading...")).toBeInTheDocument();
  });

  it("supports style prop", () => {
    render(
      <Skeleton
        data-testid="skeleton"
        style={{ width: "100px", height: "20px" }}
      />,
    );

    const skeleton = screen.getByTestId("skeleton");
    expect(skeleton).toHaveStyle({ width: "100px", height: "20px" });
  });

  it("supports onClick handler", async () => {
    const handleClick = vi.fn();
    const user = userEvent.setup();

    render(<Skeleton data-testid="skeleton" onClick={handleClick} />);

    const skeleton = screen.getByTestId("skeleton");
    await user.click(skeleton);

    expect(handleClick).toHaveBeenCalledTimes(1);
  });
});
