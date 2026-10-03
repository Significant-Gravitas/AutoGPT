# **How to Download and Import an Agent from the AutoGPT Marketplace (Local Hosting)**

## **Overview**
This guide explains how to download an agent from the AutoGPT marketplace and import it into your locally hosted instance.

<center><iframe width="560" height="315" src="https://www.youtube.com/embed/eTg2kbJdBIw?si=v1npcr8HDiInvUPw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe></center>

## **Prerequisites**
* A running self-hosted AutoGPT instance
* Access to the public marketplace at [platform.agpt.co/marketplace](https://platform.agpt.co/marketplace)

## **Step-by-Step Process**

### **1. Download the Agent**
1. Open the public marketplace and click the agent you want
2. Under "Want to use this agent locally?", click **Download here**
3. The agent file (JSON) saves to your computer

### **2. Import the Agent**
1. In your self-hosted instance, open **Agents**
2. Click **Import**
3. On the **AutoGPT agent** tab, select the downloaded agent file
4. Check the agent name and description
5. Click **Upload**

### **3. Verify Import**
* The agent opens in the Builder
* It also appears in your **Agents** library

##  **Important Notes**
* Add any credentials the agent's blocks need under **Settings → Integrations** before running it