import { toast } from "@/components/molecules/Toast/use-toast";
import { isKey } from "@/lib/keyboard";
import { useState } from "react";

interface Args {
  onSend: (text: string) => Promise<void>;
}

export function useCompactComposer({ onSend }: Args) {
  const [value, setValue] = useState("");
  const canSend = value.trim().length > 0;

  async function submit() {
    const text = value.trim();
    if (!text) return;
    setValue("");
    try {
      await onSend(text);
    } catch {
      setValue(text);
      toast({
        title: "Message not sent",
        description: "Something went wrong. Your message is back in the box.",
        variant: "destructive",
      });
    }
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (!isKey(event, "Enter") || event.shiftKey) return;
    event.preventDefault();
    void submit();
  }

  return { value, setValue, canSend, submit, handleKeyDown };
}
