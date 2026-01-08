# AutoGPT Support Manual

This guide is designed to help you assist users effectively, troubleshooting technical issues while acting as a "Human CoPilot" to guide them toward value.

## 🧠 Section 1: The "Human CoPilot" Philosophy

### Q: Who is our core user, and what do they need?

Our core persona is an overwhelmed small business owner or decision-maker. They are juggling many tools and running too many processes themselves.

They need: Speed, clarity, and specific solutions to their problems.

They do NOT need: Open-ended exploration or vague advice.

Key Insight: If we don't help them with their specific goal quickly, they will not return. First impressions define trust.

### Q: The user says, "I don't know where to start" or seems overwhelmed. How should I handle this?

This is a common friction point. Users often find the platform powerful but confusing. You need to act as a Human CoPilot.
Follow this 4-step process:

Listen: Ask about the specific process they want to automate.

Recommend: Suggest the exact right agent for their situation.

Show: Share sample outputs or a demo so they can witness the capability.

Explain: Give just enough guidance to move forward (e.g., "Head to this page and set up the agent...").
Stat: Users who were initially lost often found value quickly when guided in real-time.

### Q: The user wants to automate a complex task that isn't a good fit for AI. What do I say?

Be honest. Product quality is non-negotiable. When relevance isn't possible, clarity is quality.

Do not: Let them build a bad agent that fails. This destroys trust.

Do: Explicitly state the limits and provide a clear alternative.

Script: "I’d recommend not automating that [specific task], it’s not a great fit for automation right now because [reason]. However, [Task B] is ideal and could save you a lot of time. Let's look at that."

## 🛠️ Section 2: Installation & Technical Troubleshooting

### Q: How do I install AutoGPT locally?

For most users, recommend the Auto Setup Script:

MacOS/Linux: curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh

Windows (PowerShell): powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"

Manual Installation Prerequisites:

Node.js & NPM

Docker & Docker Compose

Git

Steps: Clone the repo, copy .env.default to .env in the autogpt_platform folder, and run docker compose up -d --build.

### Q: A Windows user has an "Unhealthy" database container. What is the fix?

This is usually caused by using Hyper-V instead of WSL 2.
Steps to fix:

Install WSL 2.

Open Docker Desktop > Settings > General.

Check the box for "Use the WSL 2 based engine".

Restart Docker Desktop.

### Q: A Raspberry Pi 5 user reports the supabase-vector container is failing. How do I fix it?

The Pi 5 uses a 16K page size by default, but the container expects 4K.
Steps to fix:

Edit /boot/firmware/config.txt.

Add the line: kernel=kernel8.img

Reboot the device.

### Q: How can I verify if the application is running correctly?

Instruct the user to visit http://localhost:3000 in their browser.
Default Ports:

Frontend UI: 3000

Backend WebSocket: 8001

Execution API (REST): 8006

## 🤖 Section 3: Building & Managing Agents

### Q: How do I guide a user to create a basic Q&A agent?

Walk them through these specific steps:

Add Blocks: Drag in an Input Block, an AI Text Generator Block, and an Output Block.

Connect: Link Input → AI Prompt, and AI Response → Output Value.

Name: Name the input "question" and the output "answer".

Save & Run: Click Save, name the agent, then click Run to test.

### Q: How does a user edit an existing agent?

Go to the Monitor Tab.

Click the agent's name.

Click the Pencil Icon to enter the editor.

Modify and save.

### Q: How does a user delete an agent?

Navigate to the Monitor Tab.

Select the agent.

Click the Trash Icon on the right.

Confirm deletion (Note: This cannot be undone).

### Q: A user wants to perform calculations. How do I set that up?

Use the Calculator Block structure:

Add two Input Blocks (name them "a" and "b").

Add a Calculator Block.

Add an Output Block (name it "results").

Connect "a" and "b" inputs to the calculator's respective inputs.

Connect the calculator result to the output block.

## 📦 Section 4: Marketplace & Sharing

### Q: How does a user import an agent from the Marketplace (Local Hosting)?

Note: Local hosting interface differs from cloud.

Go to the Marketplace section.

Click "Download Agent" on the desired agent (saves a file to computer).

Go to the Monitor Tab.

In the "Create" dropdown, select "Import from File".

Upload the file and click "Import and Edit".

### Q: How does a user submit their agent to the Marketplace?

Navigate to the Marketplace and click "Submit Agent".

Select the saved agent from the dropdown.

Fill in the Description, Author Name, Keywords, and Category.

Submit for review (it will be in a "pending" state until approved by the AutoGPT team).

## ⚠️ Section 5: What AI Can & Cannot Do

### Q: What should users expect regarding AI capability?

Users often expect a "magic button" but need a "competent partner."

Can Do: Automate repetitive processes, handle specific defined tasks (bookkeeping, SEO traffic analysis), and follow clear instructions.

Cannot Do: Guess open-ended objectives without context, perfectly replicate human intuition in undefined scenarios, or function without clear "Do it for me" steps.

Guidance: Users expect Co-Pilot to provide bespoke automation expertise based on their business situation. If an agent isn't working, check if the objective is too vague or if the agent type is mismatched to the goal.
