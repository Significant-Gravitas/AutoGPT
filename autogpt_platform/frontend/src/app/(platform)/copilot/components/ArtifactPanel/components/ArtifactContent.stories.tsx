import type { Meta, StoryObj } from "@storybook/nextjs";
import { http, HttpResponse } from "msw";
import { ArtifactContent } from "./ArtifactContent";
import type { ArtifactRef } from "../../../store";
import type { ArtifactClassification } from "../helpers";
import {
  Code,
  File,
  FileHtml,
  FileText,
  Image,
  Table,
} from "@phosphor-icons/react";

const PROXY_BASE = "/api/proxy/api/workspace/files";

function makeArtifact(overrides?: Partial<ArtifactRef>): ArtifactRef {
  return {
    id: "file-001",
    title: "test.txt",
    mimeType: "text/plain",
    sourceUrl: `${PROXY_BASE}/file-001/download`,
    origin: "agent",
    ...overrides,
  };
}

function makeClassification(
  overrides?: Partial<ArtifactClassification>,
): ArtifactClassification {
  return {
    type: "text",
    icon: FileText,
    label: "Text",
    openable: true,
    hasSourceToggle: false,
    ...overrides,
  };
}

const meta: Meta<typeof ArtifactContent> = {
  title: "Copilot/ArtifactContent",
  component: ArtifactContent,
  tags: ["autodocs"],
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "Renders artifact content based on file type classification. Supports images, HTML, code, CSV, JSON, markdown, PDF, and plain text. Bug: image artifacts render as bare <img> with no loading/error states.",
      },
    },
  },
  decorators: [
    (Story) => (
      <div
        className="flex h-[500px] w-[600px] flex-col overflow-hidden border border-zinc-200"
        style={{ resize: "both" }}
      >
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof meta>;

export const ImageArtifactPNG: Story = {
  name: "Image (PNG) — No Loading Skeleton (Bug #1)",
  args: {
    artifact: makeArtifact({
      id: "img-png",
      title: "chart.png",
      mimeType: "image/png",
      sourceUrl: `${PROXY_BASE}/img-png/download`,
    }),
    isSourceView: false,
    classification: makeClassification({ type: "image", icon: Image }),
  },
  parameters: {
    msw: {
      handlers: [
        http.get(`${PROXY_BASE}/img-png/download`, () => {
          return HttpResponse.text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300"><rect width="400" height="300" fill="#e0e7ff"/><text x="200" y="150" text-anchor="middle" fill="#4338ca" font-size="24">PNG Placeholder</text></svg>',
            { headers: { "Content-Type": "image/svg+xml" } },
          );
        }),
      ],
    },
    docs: {
      description: {
        story:
          "**BUG:** This renders a bare `<img>` tag with no loading skeleton or error handling. Compare with WorkspaceFileRenderer which has proper Skeleton + onError states.",
      },
    },
  },
};

export const ImageArtifactSVG: Story = {
  name: "Image (SVG)",
  args: {
    artifact: makeArtifact({
      id: "img-svg",
      title: "diagram.svg",
      mimeType: "image/svg+xml",
      sourceUrl: `${PROXY_BASE}/img-svg/download`,
    }),
    isSourceView: false,
    classification: makeClassification({ type: "image", icon: Image }),
  },
  parameters: {
    msw: {
      handlers: [
        http.get(`${PROXY_BASE}/img-svg/download`, () => {
          return HttpResponse.text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300"><rect width="400" height="300" fill="#fef3c7"/><circle cx="200" cy="150" r="80" fill="#f59e0b"/><text x="200" y="155" text-anchor="middle" fill="white" font-size="20">SVG OK</text></svg>',
            { headers: { "Content-Type": "image/svg+xml" } },
          );
        }),
      ],
    },
  },
};

export const HTMLArtifact: Story = {
  name: "HTML",
  args: {
    artifact: makeArtifact({
      id: "html-001",
      title: "page.html",
      mimeType: "text/html",
      sourceUrl: `${PROXY_BASE}/html-001/download`,
    }),
    isSourceView: false,
    classification: makeClassification({
      type: "html",
      icon: FileHtml,
      label: "HTML",
      hasSourceToggle: true,
    }),
  },
  parameters: {
    msw: {
      handlers: [
        http.get(`${PROXY_BASE}/html-001/download`, () => {
          return HttpResponse.text(
            `<!DOCTYPE html>
<html>
<head><title>Artifact Preview</title></head>
<body class="p-8 font-sans">
  <h1 class="text-2xl font-bold text-indigo-600 mb-4">HTML Artifact</h1>
  <p class="text-gray-700">This is an HTML artifact rendered in a sandboxed iframe with Tailwind CSS injected.</p>
  <div class="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
    <p class="text-blue-800">Interactive content works via allow-scripts sandbox.</p>
  </div>
</body>
</html>`,
            { headers: { "Content-Type": "text/html" } },
          );
        }),
      ],
    },
  },
};

export const CodeArtifact: Story = {
  name: "Code (Python)",
  args: {
    artifact: makeArtifact({
      id: "code-001",
      title: "analysis.py",
      mimeType: "text/x-python",
      sourceUrl: `${PROXY_BASE}/code-001/download`,
    }),
    isSourceView: false,
    classification: makeClassification({
      type: "code",
      icon: Code,
      label: "Code",
    }),
  },
  parameters: {
    msw: {
      handlers: [
        http.get(`${PROXY_BASE}/code-001/download`, () => {
          return HttpResponse.text(
            `import pandas as pd
import matplotlib.pyplot as plt

def analyze_data(filepath: str) -> pd.DataFrame:
    """Load and analyze CSV data."""
    df = pd.read_csv(filepath)
    summary = df.describe()
    print(f"Loaded {len(df)} rows")
    return summary

if __name__ == "__main__":
    result = analyze_data("data.csv")
    print(result)`,
            { headers: { "Content-Type": "text/plain" } },
          );
        }),
      ],
    },
  },
};

export const CSVArtifact: Story = {
  name: "CSV (Spreadsheet)",
  args: {
    artifact: makeArtifact({
      id: "csv-001",
      title: "data.csv",
      mimeType: "text/csv",
      sourceUrl: `${PROXY_BASE}/csv-001/download`,
    }),
    isSourceView: false,
    classification: makeClassification({
      type: "csv",
      icon: Table,
      label: "Spreadsheet",
      hasSourceToggle: true,
    }),
  },
  parameters: {
    msw: {
      handlers: [
        http.get(`${PROXY_BASE}/csv-001/download`, () => {
          return HttpResponse.text(
            `Name,Age,City,Score
Alice,28,New York,92
Bob,35,San Francisco,87
Charlie,22,Chicago,95
Diana,31,Boston,88
Eve,27,Seattle,91`,
            { headers: { "Content-Type": "text/csv" } },
          );
        }),
      ],
    },
  },
};

export const JSONArtifact: Story = {
  name: "JSON (Data)",
  args: {
    artifact: makeArtifact({
      id: "json-001",
      title: "config.json",
      mimeType: "application/json",
      sourceUrl: `${PROXY_BASE}/json-001/download`,
    }),
    isSourceView: false,
    classification: makeClassification({
      type: "json",
      icon: Code,
      label: "Data",
      hasSourceToggle: true,
    }),
  },
  parameters: {
    msw: {
      handlers: [
        http.get(`${PROXY_BASE}/json-001/download`, () => {
          return HttpResponse.text(
            JSON.stringify(
              {
                name: "AutoGPT Agent",
                version: "2.0",
                capabilities: ["web_search", "code_execution", "file_io"],
                settings: { maxTokens: 4096, temperature: 0.7 },
              },
              null,
              2,
            ),
            { headers: { "Content-Type": "application/json" } },
          );
        }),
      ],
    },
  },
};

export const MarkdownArtifact: Story = {
  name: "Markdown",
  args: {
    artifact: makeArtifact({
      id: "md-001",
      title: "README.md",
      mimeType: "text/markdown",
      sourceUrl: `${PROXY_BASE}/md-001/download`,
    }),
    isSourceView: false,
    classification: makeClassification({
      type: "markdown",
      icon: FileText,
      label: "Document",
      hasSourceToggle: true,
    }),
  },
  parameters: {
    msw: {
      handlers: [
        http.get(`${PROXY_BASE}/md-001/download`, () => {
          return HttpResponse.text(
            `# Project Summary

## Overview
This is a **markdown** artifact rendered through the global renderer registry.

## Features
- Headings and paragraphs
- **Bold** and *italic* text
- Lists and code blocks

\`\`\`python
print("Hello from markdown!")
\`\`\`

> Blockquotes are also supported.`,
            { headers: { "Content-Type": "text/plain" } },
          );
        }),
      ],
    },
  },
};

export const PDFArtifact: Story = {
  name: "PDF",
  args: {
    artifact: makeArtifact({
      id: "pdf-001",
      title: "report.pdf",
      mimeType: "application/pdf",
      sourceUrl: `${PROXY_BASE}/pdf-001/download`,
    }),
    isSourceView: false,
    classification: makeClassification({
      type: "pdf",
      icon: FileText,
      label: "PDF",
    }),
  },
  parameters: {
    msw: {
      handlers: [
        http.get(`${PROXY_BASE}/pdf-001/download`, () => {
          return HttpResponse.arrayBuffer(new ArrayBuffer(100), {
            headers: { "Content-Type": "application/pdf" },
          });
        }),
      ],
    },
    docs: {
      description: {
        story:
          "PDF artifacts are rendered in an unsandboxed iframe using a blob URL (Chromium bug #413851 prevents sandboxed PDF rendering).",
      },
    },
  },
};

export const ErrorState: Story = {
  name: "Error — Failed to Load Content",
  args: {
    artifact: makeArtifact({
      id: "error-001",
      title: "old-report.html",
      mimeType: "text/html",
      sourceUrl: `${PROXY_BASE}/error-001/download`,
    }),
    isSourceView: false,
    classification: makeClassification({
      type: "html",
      icon: FileHtml,
      label: "HTML",
      hasSourceToggle: true,
    }),
  },
  parameters: {
    msw: {
      handlers: [
        http.get(`${PROXY_BASE}/error-001/download`, () => {
          return new HttpResponse(null, { status: 404 });
        }),
      ],
    },
    docs: {
      description: {
        story:
          "Shows the error state when an artifact fails to load (e.g., old/expired file returning 404). Includes a 'Try again' retry button.",
      },
    },
  },
};

export const LoadingSkeleton: Story = {
  name: "Loading State",
  args: {
    artifact: makeArtifact({
      id: "loading-001",
      title: "loading.html",
      mimeType: "text/html",
      sourceUrl: `${PROXY_BASE}/loading-001/download`,
    }),
    isSourceView: false,
    classification: makeClassification({
      type: "html",
      icon: FileHtml,
      label: "HTML",
    }),
  },
  parameters: {
    msw: {
      handlers: [
        http.get(`${PROXY_BASE}/loading-001/download`, async () => {
          // Delay response to show loading state
          await new Promise((r) => setTimeout(r, 999999));
          return HttpResponse.text("never resolves");
        }),
      ],
    },
    docs: {
      description: {
        story:
          "Shows the skeleton loading state while content is being fetched.",
      },
    },
  },
};

export const DownloadOnly: Story = {
  name: "Download Only (Binary)",
  args: {
    artifact: makeArtifact({
      id: "bin-001",
      title: "archive.zip",
      mimeType: "application/zip",
      sourceUrl: `${PROXY_BASE}/bin-001/download`,
    }),
    isSourceView: false,
    classification: makeClassification({
      type: "download-only",
      icon: File,
      label: "File",
      openable: false,
    }),
  },
  parameters: {
    docs: {
      description: {
        story:
          "Download-only files (binary, video, etc.) are not rendered inline. The ArtifactPanel shows nothing for these — they are handled by ArtifactCard with a download button.",
      },
    },
  },
};
