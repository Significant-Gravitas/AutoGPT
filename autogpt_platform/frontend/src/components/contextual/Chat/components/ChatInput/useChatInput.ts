import {
  ChangeEvent,
  FormEvent,
  KeyboardEvent,
  useEffect,
  useState,
} from "react";

interface Args {
  onSend: (message: string) => void;
  disabled?: boolean;
  maxRows?: number;
  inputId?: string;
}

export function useChatInput({
  onSend,
  disabled = false,
  maxRows = 5,
  inputId = "chat-input",
}: Args) {
  const [value, setValue] = useState("");
  const [hasMultipleLines, setHasMultipleLines] = useState(false);

  useEffect(
    function focusOnMount() {
      const textarea = document.getElementById(inputId) as HTMLTextAreaElement;
      if (textarea) textarea.focus();
    },
    [inputId],
  );

  useEffect(
    function focusWhenEnabled() {
      if (disabled) return;
      const textarea = document.getElementById(inputId) as HTMLTextAreaElement;
      if (textarea) textarea.focus();
    },
    [disabled, inputId],
  );

  useEffect(() => {
    const textarea = document.getElementById(inputId) as HTMLTextAreaElement;
    const wrapper = document.getElementById(
      `${inputId}-wrapper`,
    ) as HTMLDivElement;
    if (!textarea || !wrapper) return;

    const isEmpty = !value.trim();
    const lines = value.split("\n").length;
    const hasExplicitNewlines = lines > 1;

    const computedStyle = window.getComputedStyle(textarea);
    const lineHeight = parseInt(computedStyle.lineHeight, 10);
    const paddingTop = parseInt(computedStyle.paddingTop, 10);
    const paddingBottom = parseInt(computedStyle.paddingBottom, 10);

    const singleLinePadding = paddingTop + paddingBottom;

    textarea.style.height = "auto";
    const scrollHeight = textarea.scrollHeight;

    const singleLineHeight = lineHeight + singleLinePadding;
    const isMultiLine =
      hasExplicitNewlines || scrollHeight > singleLineHeight + 2;
    setHasMultipleLines(isMultiLine);

    if (isEmpty) {
      wrapper.style.height = `${singleLineHeight}px`;
      wrapper.style.maxHeight = "";
      textarea.style.height = `${singleLineHeight}px`;
      textarea.style.maxHeight = "";
      textarea.style.overflowY = "hidden";
      return;
    }

    if (isMultiLine) {
      const wrapperMaxHeight = 196;
      const currentMultilinePadding = paddingTop + paddingBottom;
      const contentMaxHeight = wrapperMaxHeight - currentMultilinePadding;
      const minMultiLineHeight = lineHeight * 2 + currentMultilinePadding;
      const contentHeight = scrollHeight;
      const targetWrapperHeight = Math.min(
        Math.max(contentHeight + currentMultilinePadding, minMultiLineHeight),
        wrapperMaxHeight,
      );

      wrapper.style.height = `${targetWrapperHeight}px`;
      wrapper.style.maxHeight = `${wrapperMaxHeight}px`;
      textarea.style.height = `${contentHeight}px`;
      textarea.style.maxHeight = `${contentMaxHeight}px`;
      textarea.style.overflowY =
        contentHeight > contentMaxHeight ? "auto" : "hidden";
    } else {
      wrapper.style.height = `${singleLineHeight}px`;
      wrapper.style.maxHeight = "";
      textarea.style.height = `${singleLineHeight}px`;
      textarea.style.maxHeight = "";
      textarea.style.overflowY = "hidden";
    }
  }, [value, maxRows, inputId]);

  const handleSend = () => {
    if (disabled || !value.trim()) return;
    onSend(value.trim());
    setValue("");
    setHasMultipleLines(false);
    const textarea = document.getElementById(inputId) as HTMLTextAreaElement;
    const wrapper = document.getElementById(
      `${inputId}-wrapper`,
    ) as HTMLDivElement;
    if (textarea) {
      textarea.style.height = "auto";
    }
    if (wrapper) {
      wrapper.style.height = "";
      wrapper.style.maxHeight = "";
    }
  };

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  }

  function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    handleSend();
  }

  function handleChange(e: ChangeEvent<HTMLTextAreaElement>) {
    setValue(e.target.value);
  }

  return {
    value,
    setValue,
    handleKeyDown,
    handleSend,
    handleSubmit,
    handleChange,
    hasMultipleLines,
  };
}
