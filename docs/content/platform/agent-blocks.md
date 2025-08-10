# **How to Create an AI Agent as a Block in AutoGPT**

## **Overview**

This guide explains how to create a reusable agent block that can be used as a component in other agents.

<center><iframe width="560" height="315" src="https://www.youtube.com/embed/G5t5wbfomNE?si=dek4KKAPmx8DVOxm" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe></center>

## **What Are Agent Blocks?**

Agent blocks are pre-configured, reusable AI workflows that can be used as components within larger automation systems. Think of them as "smart building blocks" - each agent block is itself a complete workflow that can:

- Accept specific inputs
- Process data using AI and traditional automation
- Produce defined outputs
- Be easily reused in different contexts

The power of agent blocks lies in their modularity. Once you create an agent with a specific capability (like translating text or analyzing sentiment), you can reuse it as a single block in other workflows. This means you can:

- Combine multiple agent blocks to create more complex automations
- Reuse proven workflows without rebuilding them
- Share agent blocks with other users
- Create hierarchical systems where specialized agents work together

For example, a content creation workflow might combine several agent blocks:

- A research agent block that gathers information
- A writing agent block that creates the initial draft
- An editing agent block that polishes the content
- A formatting agent block that prepares the final output

## **Creating the Base Agent**

### **Required Components**

1. Input Block
2. AI Text Generator Block
3. Output Block

### **Step-by-Step Setup**

1. **Add and Configure Blocks**
    * Add an Input Block
    * Add an AI Text Generator Block
    * Add an Output Block
2. **Connect Components**
    * Connect Input's result to AI Text Generator's Prompt
    * Connect AI Text Generator's response to Output's value
3. **Name the Components**
    * Name the Input Block: "question"
    * Name the Output Block: "answer"
4. **Save the Agent**
    * Choose a descriptive name (e.g., "Weather Agent")
    * Click Save



## **Converting to a Block**

1. **Access the Block Menu**
    * Go to the Builder interface
    * Click the Blocks menu
    * Click the agent tag or search the name of your agent
2. **Using the Agent Block**
    * Click on the agent block to add to your workflow
    * Save the new agent with a descriptive name (e.g., "Weather Agent")

## **Testing the Agent Block**

1. **Run the Agent**
    * Enter a test question (e.g., "How far is the Earth from the Moon?")
    * Click Run
2. **View Results**
    * Option 1: Check "Agent Outputs" section*
    * Option 2: Click "View More" for detailed results

*Note: if there is no output block then the "Agent Outputs" button will show up blank. You can see the output under view more or at bottom of the block.

## **Advanced Usage**

* You can make more complex agents by combining multiple agent blocks
* Chain different agents together for more sophisticated workflows

## **Note**

This is a basic example that can be expanded upon to create more complex agent blocks with additional functionality.