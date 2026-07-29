export interface Persona {
  id: string;
  name: string;
  role: string;
  image: string;
  /** Mid-tone from the artwork — used for borders and the glow. */
  accent: string;
  /** Lightest tone from the artwork — used as the avatar backdrop. */
  tint: string;
}

// Colors are sampled from each SVG so the avatar's border, glow and backdrop
// stay in sync with the artwork. Autopilot is the default and has no artwork
// file — an empty image means "render the AutoGPT logo mark instead".
export const PERSONAS: Persona[] = [
  {
    id: "autopilot",
    name: "Autopilot",
    role: "Generalist",
    image: "",
    accent: "#8B5CF6",
    tint: "#F5F3FF",
  },
  {
    id: "blaze",
    name: "Blaze",
    role: "Marketer",
    image: "/personas/blaze_marketer.svg",
    accent: "#FF9E5A",
    tint: "#FFF1E8",
  },
  {
    id: "byte",
    name: "Byte",
    role: "Coder",
    image: "/personas/byte_coder.svg",
    accent: "#68CDAB",
    tint: "#E8FFF7",
  },
  {
    id: "iris",
    name: "Iris",
    role: "Designer",
    image: "/personas/iris_designer.svg",
    accent: "#E77DB1",
    tint: "#FFCEE6",
  },
  {
    id: "ledger",
    name: "Ledger",
    role: "Finance",
    image: "/personas/ledger_finance.svg",
    accent: "#BAF861",
    tint: "#F4FFE8",
  },
  {
    id: "nova",
    name: "Nova",
    role: "Researcher",
    image: "/personas/nova_researcher.svg",
    accent: "#FC5D78",
    tint: "#FFE8EC",
  },
  {
    id: "rocco",
    name: "Rocco",
    role: "Sales",
    image: "/personas/rocco_sales.svg",
    accent: "#FC5D5D",
    tint: "#FFE8E8",
  },
  {
    id: "sage",
    name: "Sage",
    role: "Writer",
    image: "/personas/sage_writer.svg",
    accent: "#7D7FE7",
    tint: "#CECFFF",
  },
  {
    id: "skye",
    name: "Skye",
    role: "Support",
    image: "/personas/skye_support.svg",
    accent: "#68AECD",
    tint: "#E8F8FF",
  },
  {
    id: "vector",
    name: "Vector",
    role: "Data Analyst",
    image: "/personas/vector_analyst.svg",
    accent: "#78ECDF",
    tint: "#CEFFFA",
  },
  {
    id: "wren",
    name: "Wren",
    role: "People Ops",
    image: "/personas/wren_hr.svg",
    accent: "#8A6FC6",
    tint: "#EFE9FE",
  },
];
