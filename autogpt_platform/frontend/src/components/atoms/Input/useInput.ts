import { InputProps } from "@/components/ui/input";
import {
  filterNumberInput,
  filterPhoneInput,
  formatAmountWithCommas,
  limitDecimalPlaces,
  removeCommas,
} from "./helpers";

interface ExtendedInputProps extends InputProps {
  decimalCount?: number;
  type?:
    | "text"
    | "email"
    | "password"
    | "number"
    | "amount"
    | "tel"
    | "url"
    | "textarea"
    | "date"
    | "datetime-local";
}

export function useInput(args: ExtendedInputProps) {
  function handleInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const { value } = e.target;
    const decimalCount = args.decimalCount ?? 4;

    let processedValue = value;

    if (args.type === "number") {
      // Basic number filtering - no decimal limiting
      const filteredValue = filterNumberInput(value);
      processedValue = filteredValue;
    } else if (args.type === "amount") {
      // Amount type with decimal limiting and comma formatting
      const cleanValue = removeCommas(value);
      let filteredValue = filterNumberInput(cleanValue);
      filteredValue = limitDecimalPlaces(filteredValue, decimalCount);

      const displayValue = formatAmountWithCommas(filteredValue);
      e.target.value = displayValue;
      processedValue = filteredValue; // Pass clean value to parent
    } else if (args.type === "tel") {
      processedValue = filterPhoneInput(value);
    }

    // Call onChange with processed value
    if (args.onChange) {
      // Only create synthetic event if we need to change the value
      if (processedValue !== value || args.type === "amount") {
        const syntheticEvent = {
          ...e,
          target: {
            ...e.target,
            value: processedValue,
          },
        } as React.ChangeEvent<HTMLInputElement>;

        args.onChange(syntheticEvent);
      } else {
        args.onChange(e);
      }
    }
  }

  function handleTextareaChange(e: React.ChangeEvent<HTMLTextAreaElement>) {
    if (args.onChange) {
      // Create synthetic event with HTMLInputElement-like target for compatibility
      const syntheticEvent = {
        ...e,
        target: {
          ...e.target,
          value: e.target.value,
        },
      } as unknown as React.ChangeEvent<HTMLInputElement>;

      args.onChange(syntheticEvent);
    }
  }

  return { handleInputChange, handleTextareaChange };
}
