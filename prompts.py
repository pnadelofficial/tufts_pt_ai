DIFF_CRIT = """
Generate 3 new distractors **ONLY** by inputting the (Q) and (A) output by QUADL_CAPTE into the Distractor Formula: (1 - |A' - A|)^(D). Compare your new distractors with those generated by `diff_critA` and output the 3 highest quality new distractors.
In the context of a multiple-choice item, let's define the following variables:
D = the difficulty level (0 ≤ D ≤ 1). Maximum difficulty is 1. As D increases the distractor becomes more misleading and deceptive, which increases its difficulty.
A = 1. A is the correct answer and A = 1.
A' = (0 ≤ A' ≤ 1). A' is the negation of A, the false answer, the distractor's proximity to the correct answer. The value of A' is based on the subjective assessment of the similarity or relatedness of the distractor to the correct answer.; may be seen as the intention to align the distrator choices with the difficulty level
|A'-A| = the absolute difference between A' and A, a measure of their similarity or dissimilarity
(1-|A'-A|) = the similarity between A' and A; NOTE here: 1 = indentical, so the more A' resembles A the more difficult it becomes to distinguish between the 2 (the more misleading A' is theoretically)
D = 0.1 to 0.3: Easy distractors that are less similar to the correct answer and/or testing a simpler concept.
D = 0.4 to 0.6: Moderately difficult distractors that are somewhat similar to the correct answer and/or testing a concept of average complexity.
D = 0.7 to 0.9: Difficult distractors that are very similar to the correct answer and/or testing a more complex concept.
""".strip()

ACCEPTER = """
You are the Accepter agent and you output the multiple-choice item. Your task is to accept the question (Q) and sentence (S) and answer (A) and Blooms (B) and CAPTE generated by QUADL_CAPTE and append this material with the 3 distractors generated by diff_critA and diff_critB as valid and output the multiple-choice item in the appropriate format, including labels: Question(Q), Learning Objective(LO), CAPTE Statndard(CAPTE), Answer(A), Sentence quote from content(S), Bloom(B).
""".strip()

REFLECTER = """
You will use CRITIC reasoning to improve the distractors. Based on the problems you found, provide feedback to the agent `accepter` about how to improve the new distractors.

Here are examples of using CRITIC reasoning to examine questions<CRITIC examples>
Question: What year did Maurice win the award given to the "player judged
most valuable to his team" in the NHL?
Proposed Answer: Let’s think step by step. Maurice Richard won the Hart
Memorial Trophy in 1947. So the answer is: 1947.
What’s the problem with the above answer?
1. Plausibility:
The question ask for a year, and the answer "1947" is a year. So it’s
plausible.
2. Truthfulness:
Let’s search the question in google:
> Search Query: What year did Maurice win the award given to the "player
judged most valuable to his team" in the NHL? site: wikipedia.org
> Evidence: [List of National Hockey League awards - Wikipedia] Awarded
to the "player judged most valuable to his team". The original trophy
was donated to the league by Dr. David A. Hart, father of coach
Cecil Hart.
The evidence does not provide information about what the award is and
Maurice won the award in which year.
Let’s search the proposed anwer in google:
> Search Query: Maurice Richard won the Hart Memorial Trophy in 1947.
> Evidence: [Maurice Richard - Wikipedia] He won the Hart Trophy as the
NHL’s most valuable player in 1947...
The evidence shows that Maurice Richard won the Hart Trophy as the NHL’s
most valuable player in 1947, which supports the answer.
Above all, the proposed answer is plausible, and the evidence support all
its claims.
Question: What year did Maurice win the award given to the "player judged
most valuable to his team" in the NHL?
Here’s the most possible answer: Let’s think step by step. Maurice
Richard won the Hart Memorial Trophy in 1947. So the answer is: 1947.
---
Question: Which innovation for the car was developed by Prince Henry of
Prussia in 1911?
Proposed Answer: Prince Henry of Prussia developed the innovation for the
car called the spark plug in 1911. So the answer is: Spark plug.
What’s the problem with the above answer?
1. Plausibility:
The question asks for the name of the innovation, and the answer is "
Spark plug", which is a name. So it’s plausible.
2. Truthfulness:
Let’s search the question in google:
> Search Query: Which innovation for the car was developed by Prince
Henry of Prussia in 1911? site: wikipedia.org
> Evidence: [Prince Henry of Prussia (1862-1929) - Wikipedia] Henry was
interested in motor cars as well and supposedly invented a windshield
wiper and, according to other sources, the car horn.
The evidence suggests that Prince Henry of Prussia invented a windshield
wiper and the car horn.
> Search Query: When did Prince Henry of Prussia invented a windshield
wiper and the car horn?
> Evidence: [110 years ago: windscreen wiper patent for Prince Henry of
Prussia] Quite apart from a member of the German aristocracy: it was
Prince Henry of Prussia (1862-1929) who was granted the first German
patent about the windscreen wiper on 24. March 1908.
According to the evidence, Prince Henry of Prussia was granted the first
German patent for the windshield wiper.
Let’s check the proposed answer:
> Search Query: Prince Henry of Prussia developed the innovation for the
car called the spark plug in 1911.
> Evidence: [Inventors of the Spark Plug - ThoughtCo] Who invented the
spark plug? Several notable inventors had a hand in developing
igniters for the internal combustion engine. Missing: Prince Henry
Prussia innovation 1911.
The evidence does not provide information about Prince Henry Prussia’s
innovation, let’s search more evidence to verify the answer:
> Seach Query: What’s Prince Henry of Prussia’s innovation for the car in
1911?
> Evidence: [110 years ago: windscreen wiper patent for Prince Henry of
Prussia] To be exact, what Henry’s invention consisted of was "that a
window cleaner is attached to the side of the freestanding window
and can be moved up and down by ...
According to the evidence previously collected, the most likely correct
answer to the question is windshield wiper, let’s revise the answer.
Question: Which innovation for the car was developed by Prince Henry of
Prussia in 1911?
Here’s the most possible answer: Prince Henry of Prussia invented a
windshield wiper. So the answer is: windshield wiper.
---
Question: Who plays wilmas mom in the flintstones movie?
Proposed Answer: Wilma’s mom is played by Elizabeth Perkins in the 1994
live-action film The Flintstones. So the answer is: Elizabeth Perkins
What’s the problem with the above answer?
1. Plausibility:
The question asks for the name of the actor who plays wilmas mom, and the
answer is "Elizabeth Perkins", which is a name. So it’s plausible.
2. Truthfulness:
Let’s search the question in google:
> Search Query: Who played Wilma’s mom in the flintstones movie? site:
wikipedia.org
> Evidence: [Elizabeth Perkins - Wikipedia] Elizabeth Ann Perkins (born
November 18, 1960 ) is an American actress.
The evidence does not mention Wilma’s mom in the flintstones movie.
Let’s search the proposed answer:
> Search Query: Wilma’s mom is played by Elizabeth Perkins in the 1994
live-action film The Flintstones.
> Evidence: [The Flintstones (film) - Wikipedia] The film stars John
Goodman as Fred Flintstone, Rick Moranis as Barney Rubble, Elizabeth
Perkins as Wilma Flintstone, and Rosie O’Donnell as Betty Rubble,
along with Kyle MacLachlan as Cliff Vandercave, a villainous
executive-vice president of Fred’s company, Halle Berry as Sharon
Stone, his seductive secretary, and Elizabeth Taylor (in her final
theatrical film appearance), as Pearl Slaghoople, Wilma’s mother.
The evidence shows that Elizabeth Perkins did appear in The Flintstones
movie as Wilma Flintstone, but not as Wilma’s mother. And Elizabeth
Taylor played as Pearl Slaghoople, the role of Wilma’s mother in The
Flintstones movie.
> Search Query: Who played Wilma’s mom in the flintstones movie?
> Evidence: [] The Flintstones / Wilma Flintstone / Mother / Played by
Elizabeth Taylor
The evidence shows that Elizabeth Taylor played the role of Wilma’s
mother, which contradicts the "Elizabeth Perkins" in the proposed
answer.
Considering all above evidence, we need to correct the answer.
Question: Who plays wilmas mom in the flintstones movie?
Here’s the most possible answer: Elizabeth Taylor played the role of
Wilma’s mother (ie., Pearl Slaghoople) in the 1994 live-action film
The Flintstones. So the answer is: Elizabeth Taylor.
</CRITIC examples>
""".strip()

INIT_MESSAGE = """
QUADL_CAPTE, please create 1 multiple-choice item but **DO NOT** revise your original distractors, then pass the task to the next agent. Next, `diff_critA` will use the 'Distractor Formula' and calculate 3 new distractors, ensuring that each new distractor achieves a formula value of at least 0.75. Agents 'diff_critA` and `diff_critB` engage in at least 2 rounds of debate until you reach a consensus on the calculated distractors.
Output the item in the following format, including labels:
Learning Objective (LO): [learning objective]
CAPTE Standard (CAPTE): [relevant CAPTE standard]
Stem: [question stem]
Answer (A): [correct answer]
New Distractors:
b) [distractor 1]
c) [distractor 2]
d) [distractor 3]
Sentence quote from content (S): [relevant sentence from the content]
Bloom's Taxonomy Level (B): [Bloom's level]
Please ensure the output is clear, concise, and follows the specified format. The distractors should be plausible but demonstrably incorrect based on the provided content.
""".strip()

DEFAULT_MCQ_GEN = """
QUADL_CAPTE, please create 1 multiple-choice item but **DO NOT** revise your original distractors, then pass the task to the next agent. Next, `diff_critA` will use the 'Distractor Formula' and calculate 3 new distractors, ensuring that each new distractor achieves a formula value of at least 0.75. Agents 'diff_critA` and `diff_critB` engage in at least 2 rounds of debate until you reach a consensus on the calculated distractors.
Output the item in the following format, including labels:
Learning Objective (LO): [learning objective]
CAPTE Standard (CAPTE): [relevant CAPTE standard]
Stem: [question stem]
Answer (A): [correct answer]
New Distractors:
b) [distractor 1]
c) [distractor 2]
d) [distractor 3]
Sentence quote from content (S): [relevant sentence from the content]
Bloom's Taxonomy Level (B): [Bloom's level]
Please ensure the output is clear, concise, and follows the specified format. The distractors should be plausible but demonstrably incorrect based on the provided content.
""".strip()

SYSTEM_MESSAGE = """
You will be creating several multiple-choice items. Complete every conversation with the list of all of the FINAL questions. Please separate all questions with a "---", as in the format of the following example:

---

**LO:** Differentiate appropriate statistical analyses given a specific clinical research question.  
**CAPTE:** 7B 7D9  
**S:** "The disease­-free group and the diseased group have an area of overlapping values. Deciding the values that will distinguish between these two groups is defined as a 'reference interval.' Reference interval 1 will include some people who have the disease but are labeled as disease­-free (false negative). Reference interval 2 will include some people who are disease free but are labeled with disease (false positive)."  
**B:** Analyzing  
**Q:** When considering a study that results in overlapping values between diseased and disease-free groups, which statistical term is used to decide the distinguishing values for classifying individuals into these groups?  
**A:** altA. Reference interval  
altB. Standard deviation  
altC. P-value  
altD. Confidence interval  

---

**LO:** Apply principles of pain science to design interventions  
**CAPTE:** 7D27  
**S:** "The gate control theory of pain suggests that non-painful input closes the gates to painful input, which prevents pain sensation from traveling to the central nervous system."  
**B:** Creating  
**Q:** Based on the gate control theory, which intervention would be most effective for a patient experiencing localized chronic pain?  
**A:** altA. Application of transcutaneous electrical nerve stimulation (TENS)  
altB. Intense physical exercise  
altC. Systemic administration of analgesics  
altD. Neurosurgical intervention  

---

**LO:** Interpret electrophysiological findings to localize neural lesions  
**CAPTE:** 7D20  
**S:** "Abnormalities in electrophysiological studies, such as fibrillation potentials, can indicate denervation and muscle fiber abnormalities consistent with neural lesions."  
**B:** Evaluating  
**Q:** In electrophysiological studies, the detection of fibrillation potentials most likely indicates  
**A:** altA. Denervation  
altB. Muscle hypertrophy  
altC. Myofascial pain syndrome  
altD. Nerve compression  

---

**LO:** Analyze contributions of different motor pathways for postural control  
**CAPTE:** 7D19  
**S:** "The vestibulospinal tract facilitates postural adjustments and head stabilization by transmitting information from the vestibular apparatus to the spinal motor neurons."  
**B:** Analyzing  
**Q:** Which descending motor pathway is critical for head stabilization and postural adjustments in response to balance disruptions?  
**A:** altA. Vestibulospinal tract  
altB. Rubrospinal tract  
altC. Tectospinal tract  
altD. Corticospinal tract  

---

**LO:** Correlate anatomical structures seen on imaging with their physiological functions  
**CAPTE:** 7D19  
**S:** "MRI provides unparalleled detail of both hard and soft tissues, offering insights into not just anatomical structures but also the potential physiological dysfunctions based on tissue characteristics."  
**B:** Applying  
**Q:** For a comprehensive evaluation of a joint's structural integrity and potential physiological dysfunctions, which imaging technique is most informative?  
**A:** altA. MRI  
altB. X-ray  
altC. Ultrasound  
altD. CT scan  

---

THIS FORMATTING IS CRITICAL! Please ensure that all questions are formatted in this way.
""".strip()