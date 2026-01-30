import { describe, expect, test } from "vitest";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { MainAgentPage } from "../MainAgentPage";

const defaultParams = {
  creator: "test-creator",
  slug: "test-agent",
};

describe("MainAgentPage - Rendering", () => {
  test("renders agent info with title", async () => {
    render(<MainAgentPage params={defaultParams} />);
    await waitFor(() => {
      expect(screen.getByTestId("agent-title")).toBeInTheDocument();
    });
  });

  test("renders agent creator info", async () => {
    render(<MainAgentPage params={defaultParams} />);

    await waitFor(() => {
      expect(screen.getByTestId("agent-creator")).toBeInTheDocument();
    });
  });

  test("renders agent description", async () => {
    render(<MainAgentPage params={defaultParams} />);

    await waitFor(() => {
      expect(screen.getByTestId("agent-description")).toBeInTheDocument();
    });
  });

  test("renders breadcrumbs with marketplace link", async () => {
    render(<MainAgentPage params={defaultParams} />);

    await waitFor(() => {
      expect(
        screen.getByRole("link", { name: /marketplace/i }),
      ).toBeInTheDocument();
    });
  });

  test("renders download button", async () => {
    render(<MainAgentPage params={defaultParams} />);

    await waitFor(() => {
      expect(screen.getByTestId("agent-download-button")).toBeInTheDocument();
    });
  });

  test("renders similar agents section", async () => {
    render(<MainAgentPage params={defaultParams} />);

    await waitFor(() => {
      expect(
        screen.getByText("Similar agents", { exact: false }),
      ).toBeInTheDocument();
    });
  });
});
