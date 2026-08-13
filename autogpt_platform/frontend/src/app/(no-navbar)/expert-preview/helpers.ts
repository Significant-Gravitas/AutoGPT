export interface Profession {
  slug: string;
  label: string;
}

export const PROFESSIONS: Profession[] = [
  { slug: "marketing_strategist", label: "Marketing Strategist" },
  { slug: "sales_development_rep", label: "Sales Development Representative" },
  { slug: "frontend_developer", label: "Frontend Developer" },
  { slug: "backend_developer", label: "Backend Developer" },
  { slug: "product_designer", label: "Product Designer" },
  { slug: "ux_researcher", label: "UX Researcher" },
  { slug: "product_manager", label: "Product Manager" },
  { slug: "data_analyst", label: "Data Analyst" },
  { slug: "ai_researcher", label: "AI Researcher" },
  { slug: "copywriter", label: "Copywriter" },
  { slug: "content_creator", label: "Content Creator" },
  { slug: "recruiter", label: "Recruiter" },
  { slug: "customer_success_manager", label: "Customer Success Manager" },
  { slug: "finance_advisor", label: "Finance Advisor" },
  { slug: "accountant", label: "Accountant" },
  { slug: "lawyer", label: "Lawyer" },
  { slug: "doctor", label: "Doctor" },
  { slug: "therapist", label: "Therapist" },
  { slug: "teacher", label: "Teacher" },
  { slug: "scientist", label: "Scientist" },
  { slug: "cybersecurity_specialist", label: "Cybersecurity Specialist" },
  { slug: "devops_engineer", label: "DevOps Engineer" },
  { slug: "qa_engineer", label: "QA Engineer" },
  { slug: "mobile_developer", label: "Mobile Developer" },
  { slug: "game_developer", label: "Game Developer" },
  { slug: "motion_designer", label: "Motion Designer" },
  { slug: "illustrator", label: "Illustrator" },
  { slug: "photographer", label: "Photographer" },
  { slug: "video_editor", label: "Video Editor" },
  { slug: "community_manager", label: "Community Manager" },
  { slug: "social_media_manager", label: "Social Media Manager" },
  { slug: "growth_marketer", label: "Growth Marketer" },
  { slug: "seo_specialist", label: "SEO Specialist" },
  { slug: "brand_strategist", label: "Brand Strategist" },
  { slug: "founder_ceo", label: "Founder / CEO" },
  { slug: "operations_manager", label: "Operations Manager" },
  { slug: "project_manager", label: "Project Manager" },
  { slug: "business_consultant", label: "Business Consultant" },
  { slug: "entrepreneur", label: "Entrepreneur" },
  { slug: "journalist", label: "Journalist" },
  { slug: "architect", label: "Architect" },
  { slug: "interior_designer", label: "Interior Designer" },
  { slug: "hr_manager", label: "HR Manager" },
  { slug: "event_planner", label: "Event Planner" },
  { slug: "ecommerce_manager", label: "E-commerce Manager" },
  { slug: "cloud_engineer", label: "Cloud Engineer" },
  { slug: "blockchain_developer", label: "Blockchain Developer" },
  { slug: "support_engineer", label: "Support Engineer" },
  { slug: "sales_engineer", label: "Sales Engineer" },
  { slug: "creative_director", label: "Creative Director" },
];

export function getProfessionImageSrc(slug: string) {
  return `/experts/professions/${slug}.webp`;
}

export function chunkIntoRows(
  professions: Profession[],
  rowCount: number,
): Profession[][] {
  const perRow = Math.ceil(professions.length / rowCount);
  return Array.from({ length: rowCount }, (_, row) =>
    professions.slice(row * perRow, (row + 1) * perRow),
  );
}
