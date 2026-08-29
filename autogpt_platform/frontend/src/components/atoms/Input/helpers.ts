export const NUMBER_REGEX = /[^0-9.-]/g;
export const PHONE_REGEX = /[^0-9\s()\+\[\]]/g;

export function formatAmountWithCommas(value: string): string {
  if (!value) return value;

  const parts = value.split(".");
  const integerPart = parts[0];
  const decimalPart = parts[1];

  // Add commas to integer part
  const formattedInteger = integerPart.replace(/\B(?=(\d{3})+(?!\d))/g, ",");

  // Check if there was a decimal point in the original value
  if (value.includes(".")) {
    return decimalPart
      ? `${formattedInteger}.${decimalPart}`
      : `${formattedInteger}.`;
  }

  return formattedInteger;
}

export function filterNumberInput(value: string): string {
  let filteredValue = value;

  // Remove all non-numeric characters except . and -
  filteredValue = value.replace(NUMBER_REGEX, "");

  // Handle multiple decimal points - keep only the first one
  const parts = filteredValue.split(".");
  if (parts.length > 2) {
    filteredValue = parts[0] + "." + parts.slice(1).join("");
  }

  // Handle minus signs - only allow at the beginning
  if (filteredValue.indexOf("-") > 0) {
    const hadMinusAtStart = value.startsWith("-");
    filteredValue = filteredValue.replace(/-/g, "");
    if (hadMinusAtStart) {
      filteredValue = "-" + filteredValue;
    }
  }

  return filteredValue;
}

export function limitDecimalPlaces(
  value: string,
  decimalCount: number,
): string {
  const [integerPart, decimalPart] = value.split(".");
  if (decimalPart && decimalPart.length > decimalCount) {
    return `${integerPart}.${decimalPart.substring(0, decimalCount)}`;
  }
  return value;
}

export function filterPhoneInput(value: string): string {
  return value.replace(PHONE_REGEX, "");
}

export function removeCommas(value: string): string {
  return value.replace(/,/g, "");
}
