import { isEmptyOrBlank } from "../src/utils/whitespace";

describe("WhiteSpace and empty string should return true", () => {
  test("Empty string should return true", () => {
    const emptyString = "";
    expect(isEmptyOrBlank(emptyString)).toEqual(true);
  })
  test("WhiteSpace string should return true", () => {
    const whiteSpaceString = "    ";
    expect(isEmptyOrBlank(whiteSpaceString)).toEqual(true);
  })
  test("NewLine should return true", () => {
    const newLineString = "\n\n";
    expect(isEmptyOrBlank(newLineString)).toEqual(true);
  })
})
