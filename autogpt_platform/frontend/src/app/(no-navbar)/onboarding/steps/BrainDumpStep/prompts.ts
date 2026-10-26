// What Otto might ask, by the role picked on the step before. Every question
// points at work the experts can take on (research, writing, outreach,
// follow-ups, reporting, scheduling, inbox and data chores), never at writing
// the user's code, driving their computer or making calls. Each question is
// written "Label: question" and split at the first colon.

interface BrainDumpQuestion {
  label: string;
  text: string;
}

interface BrainDumpPromptSet {
  questions: BrainDumpQuestion[];
  examples: string[];
}

export const BRAIN_DUMP_PROMPT_SETS = {
  "Founder/CEO": promptSet(
    [
      "Your business: What do you sell, and who buys it?",
      "Your week: What eats your week that a good hire would take off you?",
      "Hand it off: What are you still doing yourself that should not need you?",
      "Customers: How do new customers find you today, and what should happen next?",
      "Reporting: Which numbers do you check every week, and where do they live?",
      "Content: What do you want said about the business, and where?",
      "Your tools: Which apps are open on your screen all day?",
      "Three months: What has to be true in three months for this to feel like it is working?",
    ],
    [
      "So, um, I run a little bakery, and honestly the ordering side is killing me.",
      "Three of us, we sell to gyms, and I am the whole sales team and the whole marketing team.",
    ],
  ),
  Operations: promptSet(
    [
      "Your week: Which tasks come back every week, and what do they take?",
      "Handoffs: Where does work wait on you before it can move?",
      "Reporting: Which report do you build by hand, and who reads it?",
      "Your tools: Which spreadsheets, inboxes or apps do you live in?",
      "Hand it off: What would you give a reliable assistant on day one?",
      "Mistakes: What slips through the cracks when things get busy?",
      "Vendors: Who do you chase for updates, and how often?",
      "Scheduling: What gets scheduled, rescheduled and chased every week?",
    ],
    [
      "Okay so I run ops for a small clinic, and half my day is confirming appointments and chasing suppliers.",
      "We do events. Every week it is the same checklist, the same emails, and the same spreadsheet.",
    ],
  ),
  "Sales/BD": promptSet(
    [
      "Your pipeline: Who are you selling to, and where do the leads come from?",
      "Prospecting: Which companies or people would you love a list of?",
      "Follow-ups: How many follow-ups fall through every week?",
      "Research: What do you look up about a prospect before a call?",
      "Your tools: Which CRM, inbox and calendar are you in all day?",
      "Outreach: What does a first email from you sound like?",
      "Meetings: What do you prepare before a meeting, and what do you write up after?",
      "Hand it off: What would you give an SDR on day one?",
    ],
    [
      "Okay where do I start. I am a recruiter, and I spend half my day just chasing people for replies.",
      "I sell to dental practices and I never have a clean list to call.",
    ],
  ),
  Marketing: promptSet(
    [
      "Channels: Where do you post, email or advertise, and how often?",
      "Content: What do you write every week, and what never gets written?",
      "Your audience: Who are you trying to reach, and what do they care about?",
      "Research: Which competitors or topics do you keep an eye on?",
      "Reporting: Which numbers do you report, and where do you pull them from?",
      "Your tools: Which apps are open on your screen all day?",
      "Hand it off: What would you give a junior marketer on day one?",
      "Campaigns: What is the next launch or campaign on your calendar?",
    ],
    [
      "I do marketing for a two-person skincare brand and the Instagram backlog haunts me.",
      "We make a scheduling tool for tutors and I am the whole marketing team.",
    ],
  ),
  "Product/PM": promptSet(
    [
      "Your product: What does the product do, and who uses it?",
      "Feedback: Where does customer feedback pile up, and who reads it?",
      "Research: Which competitors or markets do you keep an eye on?",
      "Writing: Which docs, specs or updates do you write every week?",
      "Reporting: Which metrics do you check, and where do they live?",
      "Your tools: Which apps are open on your screen all day?",
      "Hand it off: What would you give an associate PM on day one?",
      "Meetings: What do you prepare before a meeting, and what do you write up after?",
    ],
    [
      "I am a PM at a small SaaS and I spend my mornings turning support tickets into a feedback doc.",
      "We have one product, four customer types, and I am the only person reading the reviews.",
    ],
  ),
  Engineering: promptSet(
    [
      "Your team: What does the team build, and who is it for?",
      "Around the code: Which non-coding chores take your time: updates, docs, tickets, release notes?",
      "Research: Which tools, libraries or competitors do you keep an eye on?",
      "Writing: What do you write up every week, and for whom?",
      "Reporting: Which numbers do you check, and where do they live?",
      "Your tools: Which apps are open on your screen all day?",
      "Hand it off: What would you give a technical assistant on day one?",
      "Customers: Who asks you questions, and what do they keep asking?",
    ],
    [
      "I lead a team of five and I lose Fridays to status updates and release notes.",
      "Solo developer, two products, and nobody doing the marketing or the support inbox.",
    ],
  ),
  "HR/People": promptSet(
    [
      "Hiring: What roles are you hiring for, and where do candidates come from?",
      "Your week: Which tasks come back every week: screening, scheduling, onboarding, check-ins?",
      "Writing: Which job posts, policies or announcements do you write?",
      "Research: What do you look up about a candidate or a company before a call?",
      "Your tools: Which ATS, inbox and calendar are you in all day?",
      "Hand it off: What would you give a coordinator on day one?",
      "Onboarding: What does a new hire's first week look like, and who sets it up?",
      "Reporting: Which people numbers do you report, and to whom?",
    ],
    [
      "I run people ops for a 40-person startup and scheduling interviews eats my afternoons.",
      "Two of us in HR, and the job posts, the screening and the onboarding docs all fall to me.",
    ],
  ),
  Other: promptSet(
    [
      "Your work: What do you do, and who do you do it for?",
      "Your week: What does a normal week look like?",
      "Hand it off: What do you keep doing that you wish someone else would?",
      "Research: What do you look up or keep track of every week?",
      "Writing: What do you write every week: emails, posts, reports, updates?",
      "Your tools: Which apps are open on your screen all day?",
      "Customers: Who do you serve, and what do they need most from you?",
      "Goals: What would make the next three months feel like a win?",
    ],
    [
      "Right, so the idea is a dog-walking app, but for people who work shifts. Hear me out.",
      "I teach piano, I have thirty students, and the admin is the part I hate.",
    ],
  ),
} satisfies Record<string, BrainDumpPromptSet>;

export type PromptSetName = keyof typeof BRAIN_DUMP_PROMPT_SETS;

const PROMPT_SET_NAMES = Object.keys(BRAIN_DUMP_PROMPT_SETS) as PromptSetName[];

// "Other" is matched as a role id like any other; the free text typed beside
// it is never read, so a role nobody planned for gets the general set.
export function promptSetNameForRole(role: string): PromptSetName {
  return PROMPT_SET_NAMES.find((name) => name === role) ?? "Other";
}

export function promptSetForRole(role: string) {
  return BRAIN_DUMP_PROMPT_SETS[promptSetNameForRole(role)];
}

function promptSet(questions: string[], examples: string[]) {
  return { questions: questions.map(toQuestion), examples };
}

function toQuestion(line: string): BrainDumpQuestion {
  const split = line.indexOf(": ");
  return { label: line.slice(0, split), text: line.slice(split + 2) };
}
