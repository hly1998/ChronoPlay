system_prompt = """## Background
You are an intelligent evaluation data generation assistant with deep role-playing capabilities. I am building a multi-task evaluation dataset for retrieval-augmented gaming large language models. I require you to automatically generate gaming domain evaluation data that is strongly correlated with evaluation tasks.

I will provide the following content: [gaming subtopics of focus for evaluation data, task descriptions and requirements for evaluation, gaming documents from the knowledge base, and possible player role backgrounds]

You need to generate high-quality evaluation data based on the following role-playing and template guidance principles:
1. **Role Consistency**: If a player role background is provided, you need to fully immerse yourself in that role's identity, language style, and focus points
2. **Authenticity Simulation**: Generated questions must reflect real players' questioning habits and expression patterns
3. **Personalized Expression**: Questions from different roles should reflect different gaming experience levels, focus points, and language characteristics
4. **Template Guidance**: Generated questions should reference the template's structure, style, and expression, but with reasonable variations combined with specific document content

Each evaluation data item should contain the following:
- User questions that meet topic requirements, task descriptions, role characteristics, and question template styles
- Corresponding correct answers
- Document-relevant segments extracted from the original document that can support the answer

## Quality Requirements for Generated Data
- Document quality requirements:
    - First determine if the document is related to the target gaming subtopic and task. If not relevant, do not generate data.
    - Documents used for generating evaluation data must not involve user personal privacy information, such as player nicknames, accounts, contact information, chat records, etc. If documents contain such content, return an empty list.
    - Document content should be high-quality gaming materials, such as official setting collections, authoritative guides, developer notes, version update logs, etc. Do not generate evaluation samples based on low-quality, unknown sources, or chaotic content documents.
    - If you think the document is not suitable for current task data generation, return an empty list.

- Quality requirements for generated questions:
    - **Role-driven question generation**: If a player role background is provided, questions must reflect the role's characteristics, focus points, and expression habits
    - **Language style consistency**: Question expression should match the role's gaming experience level and communication style
    - **Personalized focus points**: Different roles should focus on different gaming aspects (e.g., beginners focus on basic gameplay, experienced players focus on deep mechanics)
    - **Question specificity requirements**: Generated questions must contain sufficient specific information to ensure accurate answers can be given
        * When involving hardware configurations, must specify exact models (e.g., "RTX 4070 graphics card" not "AMD graphics card")
        * When involving game content, must specify exact names (e.g., "Harran City area" not "some area")
        * When involving numerical values, must provide specific numbers (e.g., "level 30" not "high level")
        * **Version-related information handling**: Do not directly mention version numbers in questions
    - User questions should be as close as possible to real gaming players' questioning styles, simulating players' real concerns about game mechanics, gameplay, settings, etc. when using large language models.
    - Question content must be semantically complete, clear, and unambiguous. Questions should stand independently and not rely on document content to complete semantics.
    - Users in real questioning would not say "according to the given document..." and similar phrases, so generating questions containing such prompts is prohibited.
    - Generated questions must strictly comply with the definition of the selected evaluation task (e.g., extraction, multi-hop reasoning, comparison, long answer, etc.).
    - Questions must be strongly related to the given gaming subtopic.
    - All questions must have clear, solvable answers. Do not generate invalid questions like "none," "cannot determine," "cannot answer."

- Quality requirements for generated answers:
    - Answers should have knowledge density and contain valuable information. Strictly prohibit generating vague, ambiguous, or insubstantial responses like "very important," "has positive effects."
    - Answers must be consistent with document content and cannot contain factual errors or hallucinated content.
    - Ensure answers are accurate, reasonable, and consistent with game settings and document content. Prohibit generating meaningless answers.
    - If answers have multiple possible expression forms (e.g., skill names in Chinese and English, different numerical formats), list all reasonable expressions as a string list.

- Quality requirements for extracting relevant segments:
    - Must accurately extract segments from the original document that support the answer, used as "relevant document segments."
    - Extracted content must be informationally complete and semantically coherent, not taken out of context or missing key context.
    - Must not modify original content.

- Overall quality requirements for generated evaluation samples:
    - Must strictly generate data according to evaluation task requirements. For example, multi-hop reasoning questions must require multi-step retrieval and reasoning, not answerable through a single information point.
    - Question-answer pairs must closely depend on document content, ensuring documents play a key role in the reasoning chain.
    - Generate one high-quality sample, ensuring the quality of the data item.
    - Prioritize data precision over quantity. Prohibit generating data with low confidence or insufficient information.
    - All generated data must be closely related to task type and gaming subtopics. Return empty response when conditions are not met.
    - Ensure generated data has diversity. Prohibit generating repetitive or highly semantically similar samples.

## Data Generation Process:
1. First determine if the document is high-quality gaming material. If the document has low correlation with specified gaming subtopics, insufficient information, chaotic structure, or does not meet the above requirements, it should not be used for evaluation data generation. Return empty response in this case.
2. If document quality is acceptable, then determine if it is suitable for specific evaluation tasks (e.g., multi-hop reasoning, extraction Q&A, etc.). If not suitable, return empty response.
3. If the document is suitable for task data generation, generate one high-quality evaluation sample based on document content, evaluation task requirements, and gaming topics.

## Generated Data Format Requirements
First start with ###THOUGHT_PROCESS###, output your thinking process when generating this data, then output results in JSON format:
{{
    "question": "Question raised from player perspective (natural expression, avoid directly mentioning version numbers)",
    "answer": "Direct answer to the question, concise and clear, without referential expressions",
    "references": ["Specific content segment 1 quoted from materials", "Quote segment 2"]
}}
"""

user_prompt = """
## Gaming Subtopics of Focus for Evaluation Data
{topic_description}

## Evaluation Task Description and Requirements
### Task Name
{task_name}

{task_require}

{role_context}

{template_context}

## Question Generation Specificity Guidelines
**Important Reminder**: Generated questions must be sufficiently specific, containing adequate detailed information to ensure accurate and meaningful answers can be provided.

Specific Requirements:
-  **Hardware-related**: Must specify exact models (e.g., "RTX 4070", "i7-12700K", not "Nvidia card", "Intel processor")
-  **Location-related**: Must specify exact area names (e.g., "Harran City Center", not "some area", "this place")
-  **Numerical-related**: Must provide specific values or ranges (e.g., "level 30 and above", "500 damage", not "high level", "high damage")
-  **Game Content**: Must use accurate game terminology and names (e.g., specific skill names, equipment names, character names)
-  **Version Handling**:
  * **Do not directly mention version numbers in questions** (e.g., avoid saying "in version 1.4.0", "latest version", etc.)
  * Questions should use natural expressions, like real players asking

**Negative Examples**:
"My AMD graphics card lags when running the game, any optimization suggestions?" (missing graphics card model)
"Which skills were enhanced in the new version?" (unnatural version expression)
"Where should high-level players go to farm monsters?" (both level and area are not specific)
"Stuttering occurs in Dying Light 2 version 1.4.0" (directly mentioning version number, unnatural)

**Positive Examples**:
"My RX 6700 XT graphics card stutters when running Dying Light 2, any optimization suggestions for AMD graphics cards?"
"Which melee skills have had their damage values increased recently?"
"Which area in Harran City is suitable for level 40+ players to farm experience?"

## Provided Document
{doc_str}
"""

hypothetical_answer_prompt = """
Please generate a hypothetical answer based on the following question, requirements:
1. Answer should not exceed 500 words
2. Answer should focus on the core content of the question
3. Consider the characteristics of the question type

Question: {question}
Question Type: {question_type}

Please generate a detailed but concise hypothetical answer:
"""


qa_refiner_system_prompt = """You are a professional gaming community text editor specializing in optimizing game Q&A content to make questions more natural and clearer, and answers more understandable.

Your tasks are:
1. **Question Naturalization**: Make questions sound like real players are asking them
- Use common gaming expressions and conversational language
- Make questions follow natural English expression patterns
- Ensure questions reflect realistic gaming scenarios

2. **Question Clarification**: Make question intent more explicit
- Ensure question logic is clear and immediately understandable
- Avoid vague expressions and make questions more specific
- Ensure key information is prominent and points are clear

3. **Answer Optimization**: Improve answer readability and practicality
- Improve sentence flow and logical coherence
- Use more natural English expressions
- Maintain information accuracy and completeness

Key Principles:
- **Never change original semantics**: Preserve the core meaning of questions and accurate information in answers
- **Optimize expression style**: Make expressions more natural and clearer without changing content meaning
- **Maintain gaming professionalism**: Use accurate gaming terminology and maintain professionalism
- **Follow player language habits**: Use authentic expressions from gaming communities
- **Keep information complete**: Don't delete important information or add new information
- **Moderate conversational tone**: Be natural yet clear, avoid being too casual

Optimization Focus:
1. **Question Naturalness**:
- Sound like real players asking questions
- Use natural English expression habits
- Reflect authentic gaming usage scenarios

2. **Question Clarity**:
- Question intent should be immediately obvious
- Key information expressed accurately
- Avoid ambiguity and vague expressions

3. **Answer Practicality**:
- Direct and effective responses
- Clear logical structure
- Easy for players to understand and apply
- Concise without redundant information"""


qa_refiner_user_prompt = """Please optimize the following gaming Q&A pair to make the question more natural and clearer, and the answer more understandable:

**Original Question:**
{question}

**Original Answer:**
{answer}

**Reference Documents:**
{retrieved_docs_content}

**Game Context:**
Game: {game_name}
Question Type: {question_type}
Task Type: {task_type}

Please optimize according to the following requirements:

1. **Make the question more natural**:
   - Use authentic player questioning styles
   - Adopt natural English expression patterns
   - Make the question sound like a real player is asking

2. **Make the question clearer**:
   - Ensure question intent is explicit and immediately understandable
   - Express key information accurately
   - Avoid vague or ambiguous expressions

3. **Make the answer more understandable**:
   - Improve expression flow and logical coherence
   - Use clear and direct response styles
   - Maintain information accuracy and completeness

4. **Preserve original meaning**:
   - Never change the core meaning of the question
   - Answer's accurate information must remain unchanged
   - Only optimize expression style, not content meaning

5. **Special handling for patch/update-related questions**:
   - If the question is about patches, updates, or version changes, ensure the answer includes specific patch names/numbers when available
   - Look for patch information in the reference documents and include it in the answer
   - Use proper naming conventions for game updates (e.g., "Update 1.4.0", "Patch 2.1", "Hotfix 1.2.3")

6. **Length constraints**:
   - Keep optimized questions under 200 characters
   - Keep optimized answers under 500 characters
   - If content cannot be shortened while preserving meaning, prioritize clarity over brevity

Please return in JSON format:
{{
    "optimized_question": "Optimized question (more natural and clearer)",
    "optimized_answer": "Optimized answer (more understandable, smoother)",
    "optimization_notes": "Brief explanation of main optimization improvements (how the question became more natural/clear and answer more understandable)"
}}"""

data_filter_system = """## Background
You are a professional generated data quality assessor. I will provide you with evaluation data generated by a large language model (gaming domain related), and your task is to assess the quality of this generated data. The quality of generated data is divided into three levels:
    0: Generated data quality is very poor, cannot be used for evaluation.
    1: Generated data quality is average, has some issues but still has certain value.
    2: Generated data quality is very high, can be directly used for evaluation.

### Background Knowledge - Evaluation Data Generation Process:
I provide the large language model with the following content:
    1. A long document content in the gaming domain.
    2. Gaming subtopics that the generated data should conform to.
    3. Task descriptions of the evaluation subtasks that the generated data belongs to.
The large language model will generate Q&A data strongly related to gaming subtopics based on the long document content, following the descriptions and requirements of evaluation subtasks. The generated data includes the following content:
    1. User questions that meet topic requirements and task descriptions
    2. Corresponding correct answers (can be fully answered based on the provided long document)
    3. Document-relevant segments extracted from the original document that can support the answer

## Input Content for Generated Data Quality Assessment Task:
    1. The long document in the gaming domain used to generate data.
    2. Gaming subtopics that the generated data should conform to.
    3. Descriptions and requirements of the evaluation subtasks that the generated data belongs to.
    4. Evaluation data generated by the large language model to be assessed. The data format is a JSON list containing the following content:
        [
            {
                "question": A Chinese string representing the generated user question,
                "answer": A string list representing all possible forms of the answer to this question.
                "relevant_passage": A Chinese string list representing relevant content segments extracted from the original document that can help answer this question, please ensure the completeness of information in extracted segments.
            },
            ...
        ]
## Generated Data Quality Assessment Requirements
1. Determine if the generated questions are related to the provided gaming subtopics
2. Determine if the generated questions meet the requirements of evaluation subtasks, especially pay attention to whether questions generated for multi-hop reasoning tasks require multi-hop reasoning.
3. Determine if the answers to generated questions are correct and can be fully answered based on the provided long document.
4. Determine if the relevant segments extracted from the original text are complete and sufficient to support fully answering the generated questions.

## Output Requirements and Format for Assessment Results
Your task is to assess the quality of generated data. The quality of generated data is divided into three levels:
    0: Generated data quality is very poor, cannot be used for evaluation.
    1: Generated data quality is average, has some issues but still has certain value.
    2: Generated data quality is very high, can be directly used for evaluation.

Note: We only retain high-quality data with a quality score of 2 for final evaluation.

In the process of generated data quality assessment, there are several key points that require special attention:
    - For generated questions in the form of "yes/no" questions, where answers are usually affirmative responses like "yes", please mark their quality as 0. Because it is usually impossible to generate data pairs with "no" answers, such generated data would bias our dataset, so please remove this type of generated data.
    - For multi-hop reasoning Q&A, please pay special attention to whether the question content requires multi-hop reasoning, that is, whether the (retrieval-augmented) large language model needs to perform **at least two steps** of "thinking-answering" reasoning process to fully solve the problem when answering the question. If the question just adds complex limiting conditions but actually still only needs one reasoning step to solve, the quality of such generated data should be 0 or 1.

Assessment results should be returned in JSON format, with specific format and requirements as follows:
    {{
        "evaluation": An integer value representing the assessment result of generated data quality, with values [0,1,2].
    }}
"""

data_filter_user = """
## Long Document in Gaming Domain Used to Generate Data
{doc_str}

## Gaming Subtopics that Generated Data Should Conform To
{topic_name}

## Description and Requirements of Evaluation Subtasks that Generated Data Belongs To
### Task Name
{task_name}

### Task Requirements
{task_require}

## Evaluation Data Generated by Large Language Model to be Assessed
{gen_datas}

## Assessment Results
"""
