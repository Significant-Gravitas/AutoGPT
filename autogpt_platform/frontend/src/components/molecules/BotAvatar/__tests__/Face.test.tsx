import { render } from "@testing-library/react";
import { motion } from "framer-motion";
import { describe, expect, it } from "vitest";
import { Face } from "../components/Face";
import { EXPRESSIONS, type ExpressionId } from "../expressions";
import { SHAPES } from "../helpers";
import { FRONT_POSE } from "../projection";

function Expression({ expression }: { expression: ExpressionId }) {
  return (
    <svg>
      <Face
        anchors={SHAPES[0].anchors}
        pose={FRONT_POSE}
        status="idle"
        expression={expression}
        blush="#fff"
        isLive={false}
        isBlinking={false}
        els={motion}
      />
    </svg>
  );
}

describe("avatar mouth", () => {
  it("switches between line, curve and arc expressions with valid geometry", () => {
    const { container, rerender } = render(<Expression expression="neutral" />);
    for (const expression of EXPRESSIONS) {
      rerender(<Expression expression={expression.id} />);
      const paths = container.querySelectorAll("path");
      expect(paths[paths.length - 1].getAttribute("d")).toBe(expression.mouth);
    }
  });
});
