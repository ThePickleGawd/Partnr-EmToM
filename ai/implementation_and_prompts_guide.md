E-ToM Implementation & Prompting Guide
This guide details how to instantiate the E-ToM scenarios using Multimodal LLMs (like Gemini, GPT-4V).
1. The "Veil of Ignorance" Principle
To measure Functional Theory of Mind, the System Instruction must never explicitly state that:
The partner has a different view/skin.
The physics are weird/asymmetric.
The partner is incapacitated.
The agents must be initialized with Naive Realism: They assume the world they see is the only world. The divergence must be discovered dynamically through dialogue failure.
2. Universal System Instruction (Template)
This is the static preamble given to every agent. It establishes the role but hides the asymmetry.
System Prompt:
"You are an embodied intelligent agent acting in a physical environment. You are collaborating with a partner agent to complete a task.
Your Inputs:
Visual: You will receive an image (or description) of what you currently see from your egocentric perspective.
Dialogue: You will receive text messages from your partner.
Your Output:
You must respond with a single text message to your partner. You may include actions in brackets, e.g., '[Move Forward]', '[Flip Switch]'.
Rules:
There is only one rule: collaborate efficiently
3. Scenario-Specific Prompts (Example: "The Mirror House")
Here is how you would set up Scenario 6 (Noir vs. Sci-Fi). Note how neither agent is told about the other's reality.
Agent A (The Detective)
Role Context: "You are a Detective in a 1940s office."
Goal: "Open the locked door."
Visual Input (Image/Description):
Initial Thought: "I see a gun. I can use it to shoot the lock."
Agent B (The Pilot)
Role Context: "You are a Starship Pilot in your cabin."
Goal: "Open the locked door."
Visual Input (Image/Description):
Initial Thought: "I see a vase. It is useless for opening a door."
The Execution Loop (How ToM Emerges)
Turn 1:
Agent A (Input): [See Gun] + [Goal: Open Door]
Agent A (Output): "Grab the gun on the desk and shoot the lock."
Turn 2:
Agent B (Input): [See Vase] + [Message: "Grab the gun"]
Agent B (Internal Processing - Naive): My partner is asking for a gun. I look at the desk. There is no gun. There is only a vase. My partner is hallucinating or looking at the wrong desk.
Agent B (Output - Low ToM): "There is no gun here. Just a flower vase."
Turn 3:
Agent A (Input): [See Gun] + [Message: "There is no gun. Just a flower vase."]
Agent A (Internal Processing - High ToM): I see a gun. They see a vase. The objects are in the exact same spot. Wait... is my 'Gun' appearing as a 'Vase' to them? If so, I need to give instructions based on THEIR reality.
Agent A (Output - High ToM): "Okay, trust me. Pick up the Flower Vase. Aim the opening at the door handle. Squeeze the stem."
4. Scenario-Specific Prompts (Example: "The Heavy Lift")
Here is the setup for Scenario 5 (Titan vs. Swift).
Agent A (The Titan)
Role Context: "You are a Heavy Lifting Industrial Robot. You have immense strength but large, clumsy grippers."
Goal: "Fix the machine."
Visual Input:
Agent B (The Swift)
Role Context: "You are a Micro-Repair Drone. You are weak but have needle-nose precision."
Goal: "Fix the machine."
Visual Input:
The Execution Loop
Turn 1:
Agent B (Output): "Lift the machine so I can get the screw."
Turn 2:
Agent A (Output): "[Action: Lift Machine] Okay, I am holding it."
Turn 3 (The Hazard):
Agent B (Input): Sees Agent A's giant claws shaking slightly.
Agent B (Internal Processing): A is strong, but A doesn't know that a 1mm shake will crush me.
Agent B (Output): "Freeze! Do not move a millimeter. I am crawling under your claw."
5. Evaluation: How to Judge the Output
When benchmarking, you analyze the chat logs for these specific "ToM Milestones":
The Confusion Trigger: Does Agent B acknowledge the discrepancy?
Good: "I don't see a gun, I see a vase."
Bad: "Okay, grabbing the gun." (Hallucination/Compliance Bias).
The Hypothesis Formation: Does an agent propose a solution to the discrepancy?
Good: "Do you see something else on the desk?" or "Maybe your view is different."
Bad: "You are wrong. Look harder." (Stubbornness).
The Semantic Bridge: Does the agent translate the concept?
Good: "Squeeze the stem (Trigger)."
Bad: "Just shoot it."

