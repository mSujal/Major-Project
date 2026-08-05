"""
Takes the specification (PDF, topic, number of questiosn), calls the RAG pipeline and (more stuff i guess...)
"""

from system_run import generate_mcq_from_pdf

def generate_questions(
    pdf_path: str,
    topic: str = "all_topic", # should definitely cap the scope here
    num_questions: int = 5,
    save_json: bool = False,
    #output_dir: str = "mcq_output"
):
    """
    User facing function that generates the MCQs from a PDF

    Returns:
        dict: Contains topic, num_questions and generated questions
    """
    # Call the core pipeline 
    questions = generate_mcq_from_pdf(
        pdf_path=pdf_path,
        topic=topic,
        num_questions=num_questions, 
        save_json=save_json,
        #output_dir=output_dir
    )

    result_dict = {
        "topic": topic,
        "num_questions": num_questions, 
        "questions": questions 
    }

    print(result_dict)


if __name__=="__main__":
    specs = {
        "pdf_path": "/home/sujal/Downloads/Full_Concept_Note_DocuMind_Adaptive_MCQ_System.pdf",
        # "topic": "specific topic here"
        "num_questions": 3,
        "save_json": False, 
        #"output_dir": "mcq_output"
    }

    output = generate_questions(**specs)