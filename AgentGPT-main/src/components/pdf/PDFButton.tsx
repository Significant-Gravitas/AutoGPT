import WindowButton from "../WindowButton";
import { FaFilePdf } from "react-icons/fa";
import { pdf } from "@react-pdf/renderer";
import React, { memo } from "react";
import MyDocument from "./MyDocument";
import type { Message } from "../../types/agentTypes";

import { useTranslation } from "react-i18next";

const PDFButton = ({
  messages,
  name,
}: {
  messages: Message[];
  name: string;
}) => {
  const content = getContent(messages);

  const downloadPDF = async () => {
    const blob = await pdf(<MyDocument content={content} />).toBlob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "my-document.pdf";
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <>
      <WindowButton
        delay={0.2}
        onClick={() => {
          downloadPDF().catch(console.error);
        }}
        icon={<FaFilePdf size={12} />}
        name="PDF"
      />
    </>
  );
};

const getContent = (messages: Message[]): string => {
  const [ t ] = useTranslation();
  // Note "Thinking" messages have no `value` so they show up as new lines
  return messages
    .map((message) => {
      if (message.type == "goal") {
        return `${t('Goal: ')}${message.value}`;
      }
      if (message.type == "task") {
        return `${t('Adding Task: ')}${message.value}`;
      }
      return message.value;
    })
    .join("\n");
};

export default memo(PDFButton);
