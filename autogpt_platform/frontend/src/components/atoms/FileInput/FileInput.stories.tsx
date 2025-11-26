import type { Meta, StoryObj } from "@storybook/nextjs";
import { useState } from "react";
import { FileInput } from "./FileInput";

const meta: Meta<typeof FileInput> = {
  title: "Atoms/FileInput",
  component: FileInput,
  tags: ["autodocs"],
  parameters: {
    layout: "centered",
    docs: {
      description: {
        component:
          "File upload input with progress and removable preview.\n\nProps:\n- accept: optional MIME/extensions filter (e.g. ['image/*', '.pdf']).\n- maxFileSize: optional maximum size in bytes; larger files are rejected with an inline error.",
      },
    },
  },
  argTypes: {
    onUploadFile: { action: "upload" },
    accept: {
      control: "object",
      description:
        "Optional accept filter. Supports MIME types (image/*) and extensions (.pdf).",
    },
    maxFileSize: {
      control: "number",
      description: "Optional maximum file size in bytes.",
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

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

export const Basic: Story = {
  parameters: {
    docs: {
      description: {
        story:
          "This example accepts images or PDFs only and limits size to 5MB. Oversized or disallowed file types show an inline error and do not upload.",
      },
    },
  },
  render: function BasicStory() {
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
