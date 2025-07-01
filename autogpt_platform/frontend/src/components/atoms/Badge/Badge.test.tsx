// import { render, screen } from "@testing-library/react";
// import { describe, expect, it } from "vitest";
// import { Badge } from "./Badge";

// describe("Badge Component", () => {
//   it("renders badge with content", () => {
//     render(<Badge variant="success">Success</Badge>);

//     expect(screen.getByText("Success")).toBeInTheDocument();
//   });

//   it("applies correct variant styles", () => {
//     const { rerender } = render(<Badge variant="success">Success</Badge>);
//     let badge = screen.getByText("Success");
//     expect(badge).toHaveClass("bg-green-100", "text-green-800");

//     rerender(<Badge variant="error">Error</Badge>);
//     badge = screen.getByText("Error");
//     expect(badge).toHaveClass("bg-red-100", "text-red-800");

//     rerender(<Badge variant="info">Info</Badge>);
//     badge = screen.getByText("Info");
//     expect(badge).toHaveClass("bg-slate-100", "text-slate-800");
//   });

//   it("applies custom className", () => {
//     render(
//       <Badge variant="success" className="custom-class">
//         Success
//       </Badge>,
//     );

//     const badge = screen.getByText("Success");
//     expect(badge).toHaveClass("custom-class");
//   });

//   it("renders as span element", () => {
//     render(<Badge variant="success">Success</Badge>);

//     const badge = screen.getByText("Success");
//     expect(badge.tagName).toBe("SPAN");
//   });

//   it("renders children correctly", () => {
//     render(
//       <Badge variant="success">
//         <span>Custom</span> Content
//       </Badge>,
//     );

//     expect(screen.getByText("Custom")).toBeInTheDocument();
//     expect(screen.getByText("Content")).toBeInTheDocument();
//   });

//   it("supports all badge variants", () => {
//     const variants = ["success", "error", "info"] as const;

//     variants.forEach((variant) => {
//       const { unmount } = render(
//         <Badge variant={variant} data-testid={`badge-${variant}`}>
//           {variant}
//         </Badge>,
//       );

//       expect(screen.getByTestId(`badge-${variant}`)).toBeInTheDocument();
//       unmount();
//     });
//   });

//   it("handles long text content", () => {
//     render(
//       <Badge variant="info">
//         Very long text that should be handled properly by the component
//       </Badge>,
//     );

//     const badge = screen.getByText(/Very long text/);
//     expect(badge).toBeInTheDocument();
//     expect(badge).toHaveClass("overflow-hidden", "text-ellipsis");
//   });
// });
