/**
 * Last line of noise rejection, after the VAD's confidence and
 * minimum-length gates.
 *
 * Whisper hallucinates a small, well-known set of phrases on near-silence
 * ("Thank you.", "Thanks for watching!"); sending one starts a real turn the
 * user never asked for.
 */

const MIN_TRANSCRIPT_CHARS = 2;

export function isRejectableTranscript(transcript: string): boolean {
  const normalised = normalise(transcript);
  if (normalised.length < MIN_TRANSCRIPT_CHARS) return true;
  if (HALLUCINATIONS.has(normalised)) return true;
  return normalised.split(" ").every((word) => FILLER_WORDS.has(word));
}

function normalise(transcript: string): string {
  return transcript
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s']/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

const FILLER_WORDS = new Set([
  "uh",
  "uhh",
  "um",
  "umm",
  "hm",
  "hmm",
  "mhm",
  "mm",
  "ah",
  "aha",
  "eh",
  "er",
  "erm",
  "oh",
  "okay",
  "ok",
  "yeah",
  "yep",
  "yup",
  "nah",
  "hey",
  "hello",
  "so",
  "well",
  "like",
  "you",
  "the",
  "a",
]);

/** Whisper's documented output on silence — never a real utterance here. */
const HALLUCINATIONS = new Set([
  "thank you",
  "thanks",
  "thank you very much",
  "thanks for watching",
  "thanks for watching the video",
  "please subscribe",
  "bye",
  "bye bye",
  "goodbye",
  "subtitles by the amara org community",
  "transcription by castingwords",
]);
