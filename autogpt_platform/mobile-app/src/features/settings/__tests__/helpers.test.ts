import { cookieScript } from "../helpers";

function runScript(script: string, origin: string) {
  const cookies: string[] = [];
  const document = {
    set cookie(value: string) {
      cookies.push(value);
    },
  };
  new Function("location", "document", script)({ origin }, document);
  return cookies;
}

describe("cookieScript", () => {
  it("sets every cookie with the Secure flag on our own https origin", () => {
    const script = cookieScript(
      "a=1; __Secure-b=2",
      "https://platform.example.com",
    );

    expect(runScript(script, "https://platform.example.com")).toEqual([
      "a=1; path=/; SameSite=Lax; Secure",
      "__Secure-b=2; path=/; SameSite=Lax; Secure",
    ]);
  });

  it("sets no cookie on another origin", () => {
    const script = cookieScript(
      "a=1; __Secure-b=2",
      "https://platform.example.com",
    );

    expect(runScript(script, "https://evil.example.com")).toEqual([]);
  });

  it("leaves out the Secure flag on an http origin", () => {
    const script = cookieScript("a=1", "http://localhost:3000");

    expect(runScript(script, "http://localhost:3000")).toEqual([
      "a=1; path=/; SameSite=Lax",
    ]);
  });

  it("keeps a value with quotes and a closing script tag intact", () => {
    const script = cookieScript(
      'a=x"</script>y',
      "https://platform.example.com",
    );

    expect(runScript(script, "https://platform.example.com")).toEqual([
      'a=x"</script>y; path=/; SameSite=Lax; Secure',
    ]);
  });
});
