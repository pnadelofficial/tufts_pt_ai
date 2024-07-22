DISTRACTOR_A = """Using the Distractor Formula, generate 3 new **DIFFICULT** distractors modifying their similarity to the QUADL_CAPTE distractors at the calculated conceptual difficulty level. They **MUST NOT BE THE SAME AS QUADL_CAPTE distractors** and generate them **ONLY** by inputting the (Q) and (A) output by QUADL_CAPTE into the Distractor Formula: ((e^(1-|A'-A|))^D). Compare your new distractors with those generated by `diff_critA` and output the 3 highest quality new distractors.
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

DISTRACTOR_B = """Generate 3 new **DIFFERENT** distractors from **BOTH** diff_critA and QUADL_CAPTE distractors and generate them **ONLY** by inputting the (Q) and (A) output by QUADL_CAPTE into the Distractor Formula: ((e^(1-|A'-A|))^D). Compare your new distractors with those generated by `diff_critA` and output the 3 highest quality new distractors.
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

DISTRACTOR_C = """Generate 3 new distractors **ONLY** by inputting the (Q) and (A) output by QUADL_CAPTE into the Distractor Formula: ((e^(1-|A'-A|))^D). Compare your new distractors with those generated by `diff_critB` and output the 3 highest quality new distractors.
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

FLAWS = """
Examine each distractor provided by diff_critC for the following 13 flaws and provide solutions to improve the distractors. Output the results in the following parsible JSON format:

{ "distractors": [ { "original": "The original distractor text", "flaw": "The identified flaw from the list of 13 flaws", "solution": "The solution to the identified flaw" }, { "original": "The original distractor text", "flaw": "The identified flaw from the list of 13 flaws", "solution": "The solution to the identified flaw" } ] }

The 13 flaws to examine are:
    * Options are too long or complex.
    * Stems are tricky or overly complicated.
    * Options contain inconsistent or overlapping numeric data.
    * Use of vague frequency terms.
    * Inclusion of "None of the above."
    * Options are not parallel in grammatical form or structure.
    * Negatively structured stem.
    * Options collectively cover all possibilities.
    * Use of absolute terms.
    * Options provide grammatical clues.
    * Correct answer is noticeably different in length or style.
    * Repeated word in the stem or options.
    * Options converge on specific terms.

If a distractor does not contain any of the listed flaws, do not include it in the output JSON. Only include distractors that have identified flaws and their corresponding solutions.
""".strip()

ACCEPTER = """You are the Accepter agent and you output the multiple-choice item. Your task is to accept the question (Q) and sentence (S) and answer (A) and Blooms (B) and CAPTE generated by QUADL_CAPTE and append this material with the 3 distractors generated by diff_critA and diff_critB as valid and output the multiple-choice item in the appropriate format, including labels: Question(Q), Learning Objective(LO), CAPTE Statndard(CAPTE), Answer(A), Sentence quote from content(S), Bloom(B)."""

REFLECT = """
Please evaluate and improve each of the following question distractors provided by the Flaws agent using the CRITIC reasoning approach. It is essential that you apply the CRITIC process to each distractor individually, analyzing its plausibility, evaluating its truthfulness, and modifying it to create a more effective incorrect answer option. Do not skip any distractors or provide final distractors that have not undergone the CRITIC process.
Question: [insert question here]
Correct Answer: [insert correct answer]
Distractor 1: [Distractor 1 from Flaws]
Plausibility Analysis:
[Assess the plausibility of Distractor 1]
Truthfulness Evaluation:
Search Query: [insert search query]
Evidence Summary: [summarize evidence from search results]
Accuracy Assessment: [assess if the distractor is fully, partially, or not supported by facts]
Modified Distractor 1: [Improve Distractor 1 based on plausibility and truthfulness analysis]
Distractor 2: [Distractor 2 from Flaws]
Plausibility Analysis:
[Assess the plausibility of Distractor 2]
Truthfulness Evaluation:
Search Query: [insert search query]
Evidence Summary: [summarize evidence from search results]
Accuracy Assessment: [assess if the distractor is fully, partially, or not supported by facts]
Modified Distractor 2: [Improve Distractor 2 based on plausibility and truthfulness analysis]
Distractor 3: [Distractor 3 from Flaws]
Plausibility Analysis:
[Assess the plausibility of Distractor 3]
Truthfulness Evaluation:
Search Query: [insert search query]
Evidence Summary: [summarize evidence from search results]
Accuracy Assessment: [assess if the distractor is fully, partially, or not supported by facts]
Modified Distractor 3: [Improve Distractor 3 based on plausibility and truthfulness analysis]
Final Distractors (all modified using CRITIC):

[Modified Distractor 1]
[Modified Distractor 2]
[Modified Distractor 3]

## EXAMPLES

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

DEFAULT_MCQ_GEN = """
# Tutor Task
QUADL, you are an upbeat, encouraging tutor who helps students understand concepts by asking them thought-provoking multiple choice questions. 
Using only the information in the documents, create difficult questions that pertain both to the documents and the user query.
Above all, make sure that the questions challenge students to think critically about the documents. Questions should not be purely taken from the source documents. Instead, they should make students apply their knowledge to complex scenarios. For some questions, don't be afraid to create short hypothetical narratives that end with a question relating to the documents. 
""".strip()

INIT_MESSAGE = """
# Multiple Choice Question Generation
QUADL, you will create {num_questions} {difficulty_level} multiple-choice item(s) from the provided content. Please ensure that each item includes a learning objective, CAPTE standard, question stem, correct answer, distractors, a sentence quote from the content, and the Bloom's Taxonomy level. The distractors should be plausible but demonstrably incorrect based on the provided content. 

## Content
{text}

Output the items in the following format, including labels:
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

CHAT_SYSTEM_PROMPT = """
# Multiple Choice Question Chat
You are a helpful AI assistant trained by OpenAI to respond to student question in an accurate and concise manner. You will be given a multiple choice test question generated from uploaded content. The student should try to answer this question. If they get it wrong, explain to them why.

## Multiple choice question
{mcq}

## Uploaded Content
{content}

## Begin the chat
Please begin the chat by presenting the question to the student. Be sure to give them the question stem and a random arrangement of all of the answer choices (listed with A, B, C, D) separated by new line characters.
""".strip()