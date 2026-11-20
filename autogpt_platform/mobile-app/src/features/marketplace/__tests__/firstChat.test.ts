import { awaitFirstChat } from "../firstChat";

function steps(found: (string | null)[]) {
  const answers = [...found];
  return {
    findLatest: jest.fn(async () => answers.shift() ?? null),
    create: jest.fn(async () => "created"),
    wait: jest.fn(async () => undefined),
  };
}

describe("awaitFirstChat", () => {
  it("opens the chat the server made, without creating one", async () => {
    const s = steps([null, null, "kickoff"]);
    await expect(awaitFirstChat({ ...s, polls: 5 })).resolves.toBe("kickoff");
    expect(s.create).not.toHaveBeenCalled();
    expect(s.wait).toHaveBeenCalledTimes(2);
  });

  it("creates a chat only when the server's never comes", async () => {
    const s = steps([]);
    await expect(awaitFirstChat({ ...s, polls: 3 })).resolves.toBe("created");
    expect(s.findLatest).toHaveBeenCalledTimes(3);
    expect(s.wait).toHaveBeenCalledTimes(2);
  });
});
