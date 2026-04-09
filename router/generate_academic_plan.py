import os
from fastapi import APIRouter,HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List

from langchain_community.vectorstores import PGVector
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from main import get_vector_store

router = APIRouter(prefix="/api/v1", tags=["Lesson Plan Generation"])

# --- New Pydantic Schema for Academic Planning ---
class AssessmentCheckpoint(BaseModel):
    checkpoint: str = Field(description="A specific moment or question during the lesson to check understanding")
    expected_response: str = Field(description="What the student should ideally understand at this point")

class AcademicPlanning(BaseModel):
    learning_objectives: List[str] = Field(description="List of specific 'Students will be able to...' goals")
    teaching_methodology: str = Field(description="The pedagogical approach (e.g., 5E Model, Inquiry-based)")
    classroom_activities: List[str] = Field(description="Step-by-step activities to be performed in class")
    assessment_checkpoint: List[AssessmentCheckpoint] = Field(description="Mid-lesson checkpoints for the teacher")
    worksheet_content: List[str] = Field(description="5 complex, application-based questions for the worksheet")
    simplified_explanation: str = Field(description="A simplified version of the most complex concept in the chapter")

# --- Endpoint Logic ---
@router.post('/generate_academic_plan', response_model=AcademicPlanning)
async def generate_academic_plan(
    class_name: str,
    subject: str,
    chapter_name: str,
    vs: PGVector = Depends(get_vector_store)
):
    # 1. RAG Retrieval
    retriever = vs.as_retriever(
        search_kwargs={"k": 20, "filter": {"chapter_name": chapter_name}}
    )

    docs = retriever.invoke(f"query: Comprehensive lesson plan, activities and objectives for {chapter_name}")
    context_text = "\n\n".join([doc.page_content for doc in docs])

    if not context_text:
        raise HTTPException(status_code=404, detail="No chapter data found in Vector DB.")

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key or "your-deepseek-api-key" in deepseek_api_key:
        raise HTTPException(status_code=500, detail="CRITICAL: DEEPSEEK_API_KEY is missing or invalid in .env file.")

    # 2. Initialize LLM & Parser
    parser = PydanticOutputParser(pydantic_object=AcademicPlanning)


    llm = ChatOpenAI(
        model="deepseek-chat", 
        api_key=deepseek_api_key, 
        base_url="https://api.deepseek.com",
        temperature=0.3
    )

    
    # 3. Create the Prompt (Integrating Pedagogy Instructions)
    system_instruction = """
    You are an Expert Pedagogy Consultant for the Indian Education System.
    Based ONLY on the provided NCERT context, create a detailed Academic Plan.
    
    REQUIRED STRUCTURE & RULES:
    1. Learning Objectives: Focus on Bloom's Taxonomy (Remember, Understand, Apply, Analyze, Evaluate, Create).
    2. Teaching Methodology: Suggest a strategy like 'Activity-Based Learning' or the '5E Model'.
    3. Classroom Activities: Practical steps the teacher can take.
    4. Assessment Checkpoints: Questions to ask *during* the flow of the class to check understanding.
    5. Worksheets: Provide 5 application-oriented questions.
    
    If the context doesn't contain a specific activity, design one that is strictly relevant to the textbook concepts.
    
    {format_instructions}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", "Context: {context}\n\nGenerate the Academic Plan for Class {class_name} {subject}: {chapter_name}.")
    ])


    chain = prompt | llm | parser

    try:
        result = chain.invoke({
            "context": context_text,
            "class_name": class_name, 
            "subject": subject,
            "chapter_name": chapter_name,
            "format_instructions": parser.get_format_instructions()
})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Processing Error: {str(e)}")