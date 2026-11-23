import { Share08Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";

import { Button } from "@/components/Button";

import { ShareSheet } from "./ShareSheet";

interface Props {
  sessionID: string;
}

export function ShareChatButton({ sessionID }: Props) {
  const [isOpen, setIsOpen] = useState(false);

  function open() {
    setIsOpen(true);
  }

  function close() {
    setIsOpen(false);
  }

  return (
    <>
      <Button
        label="Share chat"
        onPress={open}
        variant="secondary"
        size="small"
        leadingIcon={Share08Icon}
      />
      <ShareSheet sessionID={sessionID} isOpen={isOpen} onClose={close} />
    </>
  );
}
