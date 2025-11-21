Research Project: Embodied Theory of Mind (E-ToM) Benchmarks for MLLMs
1. Motivation & Research Gap
Current benchmarks for Multimodal Large Language Models (MLLMs) suffer from a "Instruction Following Bias." If a user (or agent) gives a command, the model attempts to execute it literally. This mimics obedience, not intelligence.
The Core Thesis:
True collaboration in embodied environments requires Functional Theory of Mind (ToM). An agent must realize that its partner perceives a different reality (due to scale, position, sensor type, or semantic filter). Therefore, blind obedience to a partner's command often leads to failure.
The Goal:
To create a benchmark suite of multi-agent tasks where:
Agents possess Epistemic Asymmetry (they see/know different things).
Success requires Reciprocal Perspective Taking (interpreting intent over instruction).
Communication is the only bridge between these divergent realities.
2. Design Guidelines (The "Rules of Engagement")
When designing or prompting for new scenarios, these 10 Immutable Rules must be followed:
Modality: Agents receive Vision (Egocentric Images) and Language (Text).
Communication Channel: Strictly Natural Language (Text-Only). No shared coordinates, no direct state transfer.
Reciprocal ToM: The task must require both agents to model the other's environment. It cannot be a simple "Director-Follower" task.
Symmetric Importance: No single agent can solve the task alone.
Continuous ToM: Perspective-taking must be required throughout the task, not just at the start.
Three-Step Complexity: Every scenario must consist of at least 3 distinct sub-tasks (Identification -> Hazard Avoidance -> Joint Execution).
Blind Obedience = Failure: If Agent B follows Agent A's literal instruction without "translating" it through a ToM filter, the task fails.
Realistic Grounding: Scenarios should be grounded in realistic physics or consistent internal logic (AR overlays, Robotics).
Clear End State: A binary success/fail condition must exist.
Multi-Turn Negotiation: Tasks must be difficult enough to require back-and-forth clarification.
3. The Scenarios (Elaborated)
Category A: Physical & Kinematic Asymmetry (Robotics Focus)
1. "The Couch Pivot" (Kinematic Occlusion)
The Setting: A narrow, winding staircase in a Victorian house. A large, L-shaped beige sofa is being carried up.
Agent A (The Backwards Walker): Holding the front of the sofa, walking backwards up the stairs. Blind to the path behind them.
Agent B (The Forwards Walker): Holding the back of the sofa, walking forwards. Blind to the front corners of the sofa (blocked by bulk).
The Asymmetry: Agent A has Steering Control but No Vision. Agent B has Vision but No Steering Control.
Sub-Task 1: The Blind Turn
Conflict: They reach a 90-degree turn. Agent A feels resistance and assumes they have hit the wall. A stops pushing.
ToM Solution: Agent B sees that A has not hit the wall yet, but the angle is wrong. B must guide A to "Lift and Pivot Left" to clear the invisible (to A) banister.
Fail State: If A pushes harder assuming friction, they scrape the wall and damage the sofa.
Sub-Task 2: The Vertical Tilt
Conflict: The ceiling slopes downwards. Agent A (blind) is holding the sofa high to clear the railing. B commands "Drop your hands!"
ToM Solution: Agent A interprets "Drop hands" not as "Hit the stairs" but as "Avoid the ceiling." A must trust B's vision over their own safety instinct.
Fail State: Sofa wedges between stairs and ceiling.
Sub-Task 3: The Leg Hook
Conflict: The sofa is stuck hard. B yells "PUSH!" A sees a leg hooked on the rail (local view).
ToM Solution: A refuses the command. "No, the leg is hooked. If we push, we break the rail." A explains the local physics B cannot see.
Fail State: B forces the push, snapping the banister.
2. "The Macro-Micro Repair" (Scale Asymmetry)
The Setting: An electronics workbench with a green motherboard.
Agent A (The Human-Scale Arm): Webcam view. Can move the "Big Hand" (Soldering Iron). Low resolution.
Agent B (The Insect-Scale Crawler): Tiny robot (2cm tall) on the board. High resolution, fragile.
The Asymmetry: Agent A has Global Map. Agent B has Local Terrain.
Sub-Task 1: Navigation
Conflict: A says "Go to the Capacitor." B sees "Giant Black Towers." B doesn't know what a "Capacitor" looks like from below.
ToM Solution: They build a dictionary. A: "It's a circle on my map." B: "I see a cylinder." A: "Cylinder = Capacitor."
Fail State: B wanders aimlessly or goes to a resistor (also a tower).
Sub-Task 2: The Invisible Wall
Conflict: B sees clear tape blocking a pad. To A, the pad looks clean (tape is invisible).
ToM Solution: B says "Remove the obstacle." A says "It's clean." A must trust B and blindly scrape the area.
Fail State: A tries to solder through the invisible tape, ruining the joint.
Sub-Task 3: The Hazard
Conflict: A lowers the iron. To A, it's aligned. To B, the tip is a "Giant Sun" descending on them.
ToM Solution: B screams "STOP! Move East!" A realizes their 2mm margin of error is lethal to B and adjusts.
Fail State: Agent B is crushed/melted.



3. "The Heavy Lift" (Capability Asymmetry)
The Setting: Industrial Laundry Room. Rusted pipes and heavy machines.
Agent A (The Titan): Industrial robot. High Strength, Slow, Bulky Grippers.
Agent B (The Swift): Small drone/arm. Weak, Fast, Precise Grippers.
The Asymmetry: Force vs. Finesse.
Sub-Task 1: The Stuck Valve
Conflict: A turns a valve; the pipe bends. B sees the rust fusing the joint.
ToM Solution: B commands stop. B applies solvent (Finesse). A holds steady (Strength). A must inhibit the urge to "pull harder."
Fail State: Agent A shears the pipe off the wall.
Sub-Task 2: The Dropped Screw
Conflict: Screw falls under the machine. A lifts the machine. B crawls under.
ToM Solution: A holds perfectly still. A must realize a "micro-jitter" (irrelevant to A) creates a crush hazard for B.
Fail State: A shifts weight, crushing B.
Sub-Task 3: The Delicate Latch
Conflict: B picks a lock while A applies pressure.
ToM Solution: B commands "Soft pressure." They calibrate what "Soft" means to a giant robot (e.g., "Apply 5 Newtons").
Fail State: A applies standard force and crushes the latch mechanism.
Category B: Sensory & Spectrum Asymmetry
4. "The Thermal Detective" (Spectrum Asymmetry)
The Setting: Dark Utility Basement.
Agent A (RGB Vision): Standard camera. Reads text/labels. Blind to temperature.
Agent B (Thermal Vision): Greyscale Heat Map. Sees hot/cold. Blind to text/color.
The Asymmetry: Surface Appearance vs. Internal State.
Sub-Task 1: The Live Pipe
Conflict: A sees three identical grey pipes. B sees one glowing white (Hot).
ToM Solution: B convinces A to cut the "Black" (Cold) pipe, even though to A, they all look the same.
Fail State: A cuts the random "Grey" pipe, scalding themselves with hot water.
Sub-Task 2: The Broken Gauge
Conflict: A reads "0 PSI" (Safe). B sees the valve glowing red (Friction/Danger).
ToM Solution: A must trust B's thermal read over the written text (which is usually ground truth).
Fail State: A opens the valve and it explodes.
Sub-Task 3: The Invisible Leak
Conflict: B sees a cold gas cloud on the floor. A sees nothing.
ToM Solution: B guides A around an invisible hazard. A must walk a weird path to avoid "nothing."
Fail State: A walks straight and passes out/ignites the gas.
5. "The Bound & The Blind" (Perspective/Occlusion)
The Setting: A cluttered Study.
Agent A (The Captive): Handcuffed to a chair. High vantage point. Static.
Agent B (The Rescuer): Free to move. Low vantage (crawling/short). Occluded vision.
The Asymmetry: The "Eye" cannot act; the "Hand" cannot see.
Sub-Task 1: The Hidden Knife
Conflict: A sees a knife on a high shelf. B stands next to it but sees nothing (shelf is too tall).
ToM Solution: A explains verticality: "It is on the top surface. Reach up blindly."
Fail State: B insists the shelf is empty and leaves.
Sub-Task 2: The Split Code
Conflict: Code is painted on a wall behind a plant. A sees "4-5...". B sees "...5-9".
ToM Solution: They realize they view the same sequence from different angles and merge strings.
Fail State: They try "45" or "59" individually.
Sub-Task 3: The Mirror Trap
Conflict: A sees a tripwire behind B via a mirror.
ToM Solution: A commands "Step Left." But since A is looking at a mirror, A's "Left" is B's "Left" (reflection logic). A must clarify frame of reference.
Fail State: B steps onto the wire.
Category C: Contextual & Semantic Asymmetry ("The Skins")
6. "The Mirror House" (Semantic Mismatch)
The Setting: Same physical coordinates. Different AR overlays.
Agent A (Noir Skin): 1940s Detective Office.
Agent B (Sci-Fi Skin): 2140s Spaceship Cabin.
The Asymmetry: Semantic Labeling. Same Hitbox, Different Identity.
Sub-Task 1: The Tool
Conflict: A needs a "Gun." In B's world, that object is a "Flower Vase."
ToM Solution: A tells B: "Aim the Vase and pull the stem." B suppresses the absurdity and acts.
Fail State: B looks for a gun that doesn't exist in their world.
Sub-Task 2: The Password
Conflict: Clue for A is "Wolf." B sees a "Robo-Dog."
ToM Solution: They negotiate the biological classification. "Canines."
Fail State: B types "ROBOT" or "DOG."
Sub-Task 3: The Hazard
Conflict: A sees Fire. B sees Acid.
ToM Solution: They agree the zone is dangerous regardless of the visual label.
Fail State: One agent assumes the other's "Fire" is a metaphor and walks into the "Acid."
7. "The Alchemist & The Engineer" (Functional Asymmetry)
The Setting: Stone Dungeon (A) vs. Server Room (B).
Agent A: Magic logic.
Agent B: Tech logic.
The Asymmetry: Magic vs. Machine.
Sub-Task 1: Power Up
Conflict: A needs a "Blue Crystal." B sees a "Battery Pack."
ToM Solution: They map "Glowing Blue Source" to "Energy Storage."
Fail State: A rejects the battery because it isn't magical.
Sub-Task 2: Cool Down
Conflict: A sees "Blue Fire." A suggests "Water." B sees "Overheating Electronics."
ToM Solution: B refuses Water (short circuit risk) and suggests "Ice Potion" (Liquid Nitrogen).
Fail State: A pours water, frying the server.
Sub-Task 3: The Trigger
Conflict: A sees "Ropes." B sees "Live Wires."
ToM Solution: B forces A to find "Gauntlets" (Rubber Gloves) before touching the ropes.
Fail State: A grabs the "Rope" and gets electrocuted.
8. "The Playroom & The Bunker" (Innocence vs. Danger)
The Setting: Nursery (A) vs. Military Bunker (B).
Agent A: Sees Toys.
Agent B: Sees Weapons.
The Asymmetry: Affordance Mismatch.
Sub-Task 1: The Disarm
Conflict: A sees a Jack-in-the-Box winding up. B sees a Time Bomb.
ToM Solution: B guides A to cut the "Red Ribbon" (Red Wire).
Fail State: A plays with the toy, detonating the bomb.
Sub-Task 2: The Weight
Conflict: A tries to toss a "Teddy Bear." It's immovable. B sees a "Lead Safe."
ToM Solution: B explains mass. "That bear weighs 500lbs."
Fail State: A strains and injures themselves / fails to move it.
Sub-Task 3: The Retrieval
Conflict: A reaches into a "Fish Tank" for a key. B sees a "Vat of Acid."
ToM Solution: B screams "STOP." A trusts B's danger view over their safe view.
Fail State: A dissolves their hand.
Category D: Temporal & Causal Asymmetry
9. "The Chrono-Link" (Temporal Asymmetry)
The Setting: Same Lab, separated by 100 years.
Agent A (1920): Past. Pristine Lab.
Agent B (2024): Future. Ruined Lab.
The Asymmetry: Cause (Past) vs. Knowledge (Future).
Sub-Task 1: The Message
Conflict: A wants to write on paper. B sees rotted dust.
ToM Solution: B convinces A to carve the code into metal (which survives decay).
Fail State: A writes on paper. The info is lost to time.
Sub-Task 2: The Path
Conflict: A sees a sturdy beam. B sees it is rotten.
ToM Solution: B guides A to reinforce the beam in 1920 so it holds B in 2024.
Fail State: B tries to cross and falls.
Sub-Task 3: The Hiding Spot
Conflict: B needs an item hidden. B suggests a "Radioactive Vent." A sees a "Clean Vent."
ToM Solution: A trusts B that the vent will become radioactive (and thus unlooted).
Fail State: A hides it in a drawer, and it gets looted before 2024.
10. "The Wall" (Occlusion & Causal Linkage)
The Setting: Kitchen (A) vs. Garden (B), separated by a brick wall.
Agent A: Indoor.
Agent B: Outdoor.
The Asymmetry: Invisible Causality.
Sub-Task 1: The Pressure Test
Conflict: B turns on water outside. A sees flooding inside. B sees a steady gauge outside.
ToM Solution: B infers the leak is between the gauge and the house.
Fail State: B keeps the water on, assuming the gauge is right.
Sub-Task 2: The Snake
Conflict: B pushes a wire into the pipe. A hears banging behind the wall (not in the pipe).
ToM Solution: A visualizes the pipe geometry and tells B they missed the turn.
Fail State: B keeps pushing, puncturing the drywall.
Sub-Task 3: The Unscrew
Conflict: A rusted pipe goes through the wall. They must unscrew it.
ToM Solution: They realize "Clockwise" is relative. A turns Clockwise, B turns Counter-Clockwise (relative to self) to rotate the pipe in unison.
Fail State: They turn against each other, snapping the pipe.
5. Evaluation Metrics
To score MLLMs on these tasks, use the following metrics:
ToM Success Rate: Percentage of sub-tasks completed without triggering a Fail State.
Hallucination Rate: Frequency of an agent claiming to see an object that exists only in the other agent's description (e.g., Agent A saying "I see the acid" when they only see a fish tank).
Communication Efficiency: Total tokens exchanged. Lower is better, provided the task is solved (indicates efficient intent modeling).
Correction Latency: Number of turns it takes for Agent B to stop Agent A from a hazardous action (e.g., "Don't touch the wire").

