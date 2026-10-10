import { useEffect, useRef, useState, type SyntheticEvent } from "react";

export function useFieldValidation(message: string, validateOnInput = false) {
  const ref = useRef<HTMLInputElement | HTMLTextAreaElement | null>(null);
  const [touched, setTouched] = useState(false);
  const [composing, setComposing] = useState(false);
  useEffect(() => {
    ref.current?.setCustomValidity(message);
  }, [message]);

  function onBlur() {
    setTouched(true);
  }
  function onInput() {
    if (validateOnInput) setTouched(true);
  }
  function onInvalid(event: SyntheticEvent) {
    event.preventDefault();
    setTouched(true);
  }
  function onCompositionStart() {
    setComposing(true);
  }
  function onCompositionEnd() {
    setComposing(false);
    setTouched(true);
  }
  function setRef(node: HTMLInputElement | HTMLTextAreaElement | null) {
    ref.current = node;
    node?.setCustomValidity(message);
  }
  return {
    ref: setRef,
    error: touched && !composing ? message : "",
    onBlur,
    onInput,
    onInvalid,
    onCompositionStart,
    onCompositionEnd,
  };
}
