import { describe, expect, it } from "vitest";
import { parseCookieConsent, parseCookieConsentHeader } from "./cookiebot";

const ANSWER =
  "{stamp:'VLZnHUKBPLqZCJyClLLmnGglmUPeZsGxrmiAEZ48i7UH39ptKHY4MA==',necessary:true,preferences:false,statistics:true,marketing:false,method:'explicit',ver:1,utc:1724770548958,region:'de'}";

describe("parseCookieConsent", () => {
  it("reads each category from the stored answer", () => {
    expect(parseCookieConsent(ANSWER)).toEqual({
      necessary: true,
      preferences: false,
      statistics: true,
      marketing: false,
    });
  });

  it("reads the URL-encoded form browsers send to the server", () => {
    expect(parseCookieConsent(encodeURIComponent(ANSWER))).toEqual(
      parseCookieConsent(ANSWER),
    );
  });

  it("reads the partially encoded form Cookiebot writes", () => {
    const partial = ANSWER.replaceAll("'", "%27").replaceAll(",", "%2C");

    expect(parseCookieConsent(partial)?.statistics).toBe(true);
  });

  it('treats "-1" (no consent required in the visitor\'s region) as everything accepted', () => {
    expect(parseCookieConsent("-1")).toEqual({
      necessary: true,
      preferences: true,
      statistics: true,
      marketing: true,
    });
  });

  it("reads the answer Cookiebot stores for a region that needs no consent", () => {
    const implied =
      "{stamp:'-1',necessary:true,preferences:true,statistics:true,marketing:true,ver:2147483647,utc:1646127963403}";

    expect(parseCookieConsent(implied)?.marketing).toBe(true);
  });

  it('treats the legacy "0" answer as declined', () => {
    expect(parseCookieConsent("0")).toEqual({
      necessary: true,
      preferences: false,
      statistics: false,
      marketing: false,
    });
  });

  it("returns null when there is no answer", () => {
    expect(parseCookieConsent(undefined)).toBeNull();
    expect(parseCookieConsent(null)).toBeNull();
    expect(parseCookieConsent("")).toBeNull();
  });

  it("returns null for values it cannot read", () => {
    for (const value of ["granted", "{}", "{stamp:'x'}", "[1,2]", "%E0%A4%A"]) {
      expect({ value, parsed: parseCookieConsent(value) }).toEqual({
        value,
        parsed: null,
      });
    }
  });

  it("only grants a category on a literal true", () => {
    const parsed = parseCookieConsent(
      "{necessary:true,preferences:1,statistics:'yes',marketing:TRUE}",
    );

    expect(parsed).toEqual({
      necessary: true,
      preferences: false,
      statistics: false,
      marketing: false,
    });
  });

  it("accepts quoted booleans and extra whitespace", () => {
    const parsed = parseCookieConsent(
      " { necessary : true , statistics : \"true\" , marketing : 'true' } ",
    );

    expect(parsed?.statistics).toBe(true);
    expect(parsed?.marketing).toBe(true);
    expect(parsed?.preferences).toBe(false);
  });

  it("is not fooled by category names inside quoted values", () => {
    const parsed = parseCookieConsent(
      "{stamp:'statistics:true',necessary:true,statistics:false,marketing:false}",
    );

    expect(parsed?.statistics).toBe(false);
  });

  it("only reads top-level keys, not ones nested inside other objects", () => {
    const parsed = parseCookieConsent(
      "{stamp:'x',necessary:true,preferences:false,statistics:false,marketing:false,consentmode:{statistics:true,marketing:true},ver:1}",
    );

    expect(parsed).toEqual({
      necessary: true,
      preferences: false,
      statistics: false,
      marketing: false,
    });
  });

  it("reads top-level keys that come after a nested object", () => {
    const parsed = parseCookieConsent(
      "{consentmode:{statistics:false,list:[1,{a:2}]},necessary:true,statistics:true,marketing:false}",
    );

    expect(parsed?.statistics).toBe(true);
    expect(parsed?.marketing).toBe(false);
  });

  it("does not grant a category nested only inside another object", () => {
    expect(
      parseCookieConsent(
        "{necessary:true,preferences:false,consentmode:{statistics:true}}",
      )?.statistics,
    ).toBe(false);
  });

  it("only grants a repeated top-level key when every copy does", () => {
    expect(
      parseCookieConsent(
        "{necessary:true,statistics:false,marketing:false,statistics:true}",
      )?.statistics,
    ).toBe(false);
  });

  it("returns null when braces or quotes do not balance", () => {
    for (const value of [
      "{necessary:true,statistics:true,consentmode:{a:1}",
      "{necessary:true,statistics:true}}",
      "{stamp:'open,necessary:true,statistics:true}",
    ]) {
      expect({ value, parsed: parseCookieConsent(value) }).toEqual({
        value,
        parsed: null,
      });
    }
  });
});

describe("parseCookieConsentHeader", () => {
  const granted = encodeURIComponent(ANSWER);
  const denied = encodeURIComponent(
    ANSWER.replace("statistics:true", "statistics:false"),
  );

  it("finds the answer among other cookies", () => {
    expect(
      parseCookieConsentHeader(`a=1; CookieConsent=${granted}; b=2`)
        ?.statistics,
    ).toBe(true);
  });

  it("returns null without a CookieConsent cookie", () => {
    expect(parseCookieConsentHeader("a=1; NotCookieConsent=-1")).toBeNull();
    expect(parseCookieConsentHeader("")).toBeNull();
    expect(parseCookieConsentHeader(null)).toBeNull();
  });

  it("denies a category whichever duplicate copy denies it", () => {
    for (const header of [
      `CookieConsent=${granted}; CookieConsent=${denied}`,
      `CookieConsent=${denied}; CookieConsent=${granted}`,
    ]) {
      expect(parseCookieConsentHeader(header)?.statistics).toBe(false);
    }
  });

  it("treats an unreadable duplicate as a denial", () => {
    expect(
      parseCookieConsentHeader(`CookieConsent=-1; CookieConsent=garbage`),
    ).toEqual({
      necessary: true,
      preferences: false,
      statistics: false,
      marketing: false,
    });
  });
});
