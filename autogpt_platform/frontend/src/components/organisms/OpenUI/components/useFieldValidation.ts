import { useEffect, useRef, useState, type InvalidEvent } from "react";

export function useFieldValidation(message: string) {
  const ref = useRef<HTMLInputElement | HTMLTextAreaElement>(null);
  const [touched, setTouched] = useState(false);
  const [composing, setComposing] = useState(false);
  useEffect(() => {
    ref.current?.setCustomValidity(message);
  }, [message]);

  function onBlur() {
    setTouched(true);
  }
  function onInvalid(event: InvalidEvent<HTMLInputElement>) {
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
  return {
    ref,
    error: touched && !composing ? message : "",
    onBlur,
    onInvalid,
    onCompositionStart,
    onCompositionEnd,
  };
}
