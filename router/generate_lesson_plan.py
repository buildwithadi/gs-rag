import os
import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List

# LangChain Imports
from langchain_community.vectorstores import PGVector
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Import the lazy-loaded vector store from main
from main import get_vector_store

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize the router
router = APIRouter(prefix="/api/v1", tags=["Lesson Plan Generation"])

# --- Pydantic Models for Structured AI Output ---
class MCQ(BaseModel):
    question: str = Field(description="The multiple choice question")
    options: List[str] = Field(description="Exactly 4 options for the MCQ")
    correct_answer: str = Field(description="The correct option from the list")

class ShortAnswer(BaseModel):
    question: str = Field(description="A short answer question based on the chapter")
    answer_key: str = Field(description="The ideal correct answer for grading purposes")

class CaseBasedQuestion(BaseModel):
    scenario: str = Field(description="A real-world scenario, context, or case study extracted from the chapter text")
    question: str = Field(description="An analytical question based on the scenario")
    answer_key: str = Field(description="The ideal correct answer for grading purposes")

class LessonPlanOutput(BaseModel):
    chapter_summary: str = Field(description="A 3-4 sentence summary of the chapter")
    # Added default_factory=list so Pydantic doesn't crash if the LLM generates 0 questions
    mcqs: List[MCQ] = Field(default_factory=list, description="A list of multiple choice questions")
    short_answers: List[ShortAnswer] = Field(default_factory=list, description="A list of short answer questions")
    case_based_questions: List[CaseBasedQuestion] = Field(default_factory=list, description="A list of case-based analytical questions")

# --- Endpoints ---
@router.post("/generate_lesson_plan", response_model=LessonPlanOutput)
async def generate_lesson_plan(
    class_name: str, 
    subject: str, 
    chapter_name: str,
    num_mcqs: int = 5,
    num_short_answers: int = 3,
    num_case_based: int = 2,
    vs: PGVector = Depends(get_vector_store) # Injects the lazy-loaded vector store
):
    """ Retrieves context from Postgres and generates a structured lesson plan dynamically. """
    logger.info(f"Generating lesson plan for Class: {class_name}, Subject: {subject}, Chapter: {chapter_name}")
    logger.info(f"Requested counts -> MCQs: {num_mcqs}, Short Answers: {num_short_answers}, Case-Based: {num_case_based}")

    # 1. Vector Search (RAG)
    try:
        retriever = vs.as_retriever(
            search_kwargs={
                "k": 25, 
                "filter": {
                    "class_name": class_name,
                    "subject": subject,
                    "chapter_name": chapter_name
                }
            }
        )
        
        formatted_query = f"query: key concepts and learning objectives for {chapter_name} in {subject}"
        docs = retriever.invoke(formatted_query)
        context_text = "\n\n".join([doc.page_content for doc in docs])

        if not context_text:
            logger.warning("No context found in PostgreSQL for the given filters.")
            raise HTTPException(status_code=404, detail="No ingested data found for this class and chapter.")
        
        logger.info(f"Successfully retrieved {len(docs)} chunks from the database.")
    except Exception as e:
        logger.error(f"Database retrieval failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database retrieval failed.")

    # 2. Initialize the Standard LLM securely using .env variables
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key or "your-deepseek-api-key" in deepseek_api_key:
        logger.error("DEEPSEEK_API_KEY is missing or invalid.")
        raise HTTPException(status_code=500, detail="CRITICAL: DEEPSEEK_API_KEY is missing or invalid in .env file.")
    
    llm = ChatOpenAI(
        model="deepseek-chat", 
        api_key=deepseek_api_key, 
        base_url="https://api.deepseek.com",
        temperature=0.2
    )

    # 3. Setup the Pydantic Output Parser
    parser = PydanticOutputParser(pydantic_object=LessonPlanOutput)

    # 4. Create the Prompt with Format Instructions injected
    system_prompt = """
    You are an expert teacher. Your task is to generate a structured lesson plan and assessments 
    based strictly on the textbook context provided below. 
    Do not use outside knowledge. If the answer isn't in the context, do your best based only on the text.
    
    REQUIREMENTS:
    - Generate EXACTLY {num_mcqs} Multiple Choice Questions (MCQs).
    - Generate EXACTLY {num_short_answers} Short Answer Questions.
    - Generate EXACTLY {num_case_based} Case-Based Questions.
    
    IMPORTANT RULE: If any of the requested quantities above are 0, you MUST still include the required JSON key, but set its value to an empty array [].
    
    {format_instructions}
    
    Context from Chapter:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Generate the lesson plan and questions for Class {class_name}, Subject: {subject}, Chapter: {chapter_name}.")
    ])

    # 5. Execute the Chain (Prompt -> LLM -> Parser)
    logger.info("Sending prompt to DeepSeek API...")
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "context": context_text,
            "class_name": class_name,
            "subject": subject,
            "chapter_name": chapter_name,
            "num_mcqs": num_mcqs,
            "num_short_answers": num_short_answers,
            "num_case_based": num_case_based,
            "format_instructions": parser.get_format_instructions()
        })
        logger.info("Successfully generated and parsed JSON from DeepSeek.")
        return result
    except Exception as e:
        logger.error(f"LLM Generation or Parsing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"LLM Generation failed: {str(e)}")