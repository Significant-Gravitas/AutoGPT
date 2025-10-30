import type { Meta, StoryObj } from "@storybook/nextjs";
import { StreamingMessage } from "./StreamingMessage";
import { useEffect, useState } from "react";

const meta = {
  title: "Molecules/StreamingMessage",
  component: StreamingMessage,
  parameters: {
    layout: "padded",
  },
  tags: ["autodocs"],
} satisfies Meta<typeof StreamingMessage>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Empty: Story = {
  args: {
    chunks: [],
  },
};

export const SingleChunk: Story = {
  args: {
    chunks: ["Hello! "],
  },
};

export const MultipleChunks: Story = {
  args: {
    chunks: [
      "I can ",
      "help you ",
      "discover ",
      "and run ",
      "AI agents. ",
      "What would ",
      "you like ",
      "to do?",
    ],
  },
};

export const SimulatedStreaming: Story = {
  args: {
    chunks: [],
  },
  render: () => {
    const [chunks, setChunks] = useState<string[]>([]);
    const fullText =
      "I'm a streaming message that simulates real-time text generation. Watch as the text appears word by word, just like a real AI assistant typing out a response!";

    useEffect(function simulateStreaming() {
      const words = fullText.split(" ");
      let currentIndex = 0;

      const interval = setInterval(() => {
        if (currentIndex < words.length) {
          setChunks((prev) => [...prev, words[currentIndex] + " "]);
          currentIndex++;
        } else {
          clearInterval(interval);
        }
      }, 100); // Add a word every 100ms

      return () => clearInterval(interval);
    }, []);

    return <StreamingMessage chunks={chunks} />;
  },
};

export const LongStreaming: Story = {
  args: {
    chunks: [],
  },
  render: () => {
    const [chunks, setChunks] = useState<string[]>([]);
    const fullText =
      "This is a much longer streaming message that demonstrates how the component handles larger amounts of text. It includes multiple sentences and should wrap nicely within the message bubble. The blinking cursor at the end indicates that text is still being generated in real-time.";

    useEffect(function simulateLongStreaming() {
      const words = fullText.split(" ");
      let currentIndex = 0;

      const interval = setInterval(() => {
        if (currentIndex < words.length) {
          setChunks((prev) => [...prev, words[currentIndex] + " "]);
          currentIndex++;
        } else {
          clearInterval(interval);
        }
      }, 80);

      return () => clearInterval(interval);
    }, []);

    return <StreamingMessage chunks={chunks} />;
  },
};
