import type { Meta } from "@storybook/nextjs";
import { useState } from "react";
import { FileInput } from "./FileInput";

const meta: Meta = {
  title: "Atoms/FileInput",
  component: FileInput,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "File upload input with two variants and two modes.\n\n**Variants:**\n- `default`: Full-featured with drag & drop, progress bar, and storage note.\n- `compact`: Minimal inline design for tight spaces like node inputs.\n\n**Modes:**\n- `upload`: Uploads file to server (requires `onUploadFile` and `uploadProgress`).\n- `base64`: Converts file to base64 locally (no server upload).\n\n**Props:**\n- `accept`: optional MIME/extensions filter (e.g. ['image/*', '.pdf']).\n- `maxFileSize`: optional maximum size in bytes; larger files are rejected with an inline error.",
      },
    },
  },
};

export default meta;

function mockUpload(file: File): Promise<{
  file_name: string;
  size: number;
  content_type: string;
  file_uri: string;
}> {
  return new Promise((resolve) =>
    setTimeout(
      () =>
        resolve({
          file_name: file.name,
          size: file.size,
          content_type: file.type || "application/octet-stream",
          file_uri: URL.createObjectURL(file),
        }),
      400,
    ),
  );
}

export const Default = {
  parameters: {
    docs: {
      description: {
        story:
          "Default variant with upload mode. Full-featured with drag & drop dropzone, progress bar, and storage note. Accepts images or PDFs only and limits size to 5MB.",
      },
    },
  },
  render: function DefaultStory() {
    const [value, setValue] = useState<string>("");
    const [progress, setProgress] = useState<number>(0);

    async function onUploadFile(file: File) {
      setProgress(0);
      const interval = setInterval(() => {
        setProgress((p) => (p >= 100 ? 100 : p + 20));
      }, 80);
      const result = await mockUpload(file);
      clearInterval(interval);
      setProgress(100);
      return result;
    }

    return (
      <div className="w-[560px]">
        <FileInput
          variant="default"
          mode="upload"
          value={value}
          onChange={setValue}
          onUploadFile={onUploadFile}
          uploadProgress={progress}
          accept={["image/*", ".pdf"]}
          maxFileSize={5 * 1024 * 1024}
        />
      </div>
    );
  },
};

export const Compact = {
  parameters: {
    docs: {
      description: {
        story:
          "Compact variant with base64 mode. Minimal inline design suitable for node inputs. Converts file to base64 locally without server upload.",
      },
    },
  },
  render: function CompactStory() {
    const [value, setValue] = useState<string>("");

    return (
      <div className="w-[300px]">
        <FileInput
          variant="compact"
          mode="base64"
          value={value}
          onChange={setValue}
          placeholder="Document"
          accept={["image/*", ".pdf"]}
          maxFileSize={5 * 1024 * 1024}
        />
      </div>
    );
  },
};

export const CompactWithUpload = {
  parameters: {
    docs: {
      description: {
        story:
          "Compact variant with upload mode. Useful when you need minimal UI but still want server uploads.",
      },
    },
  },
  render: function CompactUploadStory() {
    const [value, setValue] = useState<string>("");
    const [progress, setProgress] = useState<number>(0);

    async function onUploadFile(file: File) {
      setProgress(0);
      const interval = setInterval(() => {
        setProgress((p) => (p >= 100 ? 100 : p + 20));
      }, 80);
      const result = await mockUpload(file);
      clearInterval(interval);
      setProgress(100);
      return result;
    }

    return (
      <div className="w-[300px]">
        <FileInput
          variant="compact"
          mode="upload"
          value={value}
          onChange={setValue}
          onUploadFile={onUploadFile}
          uploadProgress={progress}
          placeholder="Resume"
        />
      </div>
    );
  },
};

export const DefaultWithBase64 = {
  parameters: {
    docs: {
      description: {
        story:
          "Default variant with base64 mode. Full-featured UI but converts to base64 locally instead of uploading.",
      },
    },
  },
  render: function DefaultBase64Story() {
    const [value, setValue] = useState<string>("");

    return (
      <div className="w-[560px]">
        <FileInput
          variant="default"
          mode="base64"
          value={value}
          onChange={setValue}
          placeholder="Image"
          accept={["image/*"]}
          maxFileSize={2 * 1024 * 1024}
        />
      </div>
    );
  },
};
