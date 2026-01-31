import { describe, expect, test } from "vitest";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { MainCreatorPage } from "../MainCreatorPage";

const defaultParams = {
  creator: "test-creator",
};

describe("MainCreatorPage - Rendering", () => {
  test("renders creator info card", async () => {
    render(<MainCreatorPage params={defaultParams} />);
    await waitFor(() => {
      expect(screen.getByTestId("creator-description")).toBeInTheDocument();
    });
  });

  test("renders breadcrumbs with marketplace link", async () => {
    render(<MainCreatorPage params={defaultParams} />);

    await waitFor(() => {
      expect(
        screen.getByRole("link", { name: /marketplace/i }),
      ).toBeInTheDocument();
    });
  });

  test("renders about section", async () => {
    render(<MainCreatorPage params={defaultParams} />);

    await waitFor(() => {
      expect(screen.getByText("About")).toBeInTheDocument();
    });
  });

  test("renders agents by creator section", async () => {
    render(<MainCreatorPage params={defaultParams} />);

    await waitFor(() => {
      expect(
        screen.getByText(/Agents by/i, { exact: false }),
      ).toBeInTheDocument();
    });
  });
});
