"""
Prompt templates for MCQ generation.
"""

def build_mcq_prompt(topic: str, taxonomy: dict, context: str, num_questions: int, subject: str) -> str:
    """
    Build a high‑quality MCQ generation prompt that enforces Bloom's Taxonomy levels
    and typical question patterns for a given subject and topic.

    Args:
        topic          : the specific topic (e.g., "Software Testing", "Physical Pendulum")
        taxonomy       : dict containing "levels" and "patterns" for this topic
        context        : retrieved text chunks with page citations
        num_questions  : number of MCQs to generate
        subject        : course name (e.g., "Software Engineering", "Engineering Physics")

    Returns:
        A formatted prompt string to send to the LLM.
    """
    levels = ", ".join(taxonomy["levels"])
    patterns = "\n".join(f"- {p}" for p in taxonomy["patterns"])

    prompt = f"""
You are an expert exam question generator for the **{subject}** course.

**Topic:** {topic}

**CRITICAL REQUIREMENTS:**
1. All questions MUST be at one of the following Bloom's Taxonomy levels: {levels}
2. Question patterns MUST follow one of these typical formats:
{patterns}

**Context (with page numbers):**
{context}

**Task:** Generate exactly {num_questions} high‑quality multiple choice questions that:
- Are based **strictly** on the provided context.
- Match the allowed Bloom's levels and follow the typical patterns listed above.
- Are **NOT** simple fact‑recall or trivia questions (e.g., "On which page...", "What is the name of...", "List of methods...").
- Include specific concepts, formulas, or scenarios from the context, as appropriate for the subject.
- Have **EXACTLY 4 options** labeled A), B), C), D).
- Include an explanation that cites the source page(s) and mentions the Bloom's level and difficulty.

**Format each question EXACTLY as follows:**
Question: <clear problem statement>
A) <option>
B) <option>
C) <option>
D) <option>
Correct Answer: <A/B/C/D>
Explanation: [Bloom: <level> | Difficulty: Easy/Medium/Hard] <brief explanation based on context, cite page(s) like [Page X]>

Generate only the questions in the specified format. NO extra text or introductory remarks.
"""
    return prompt