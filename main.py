import os
from typing import TypedDict
from langchain_groq import ChatGroq # type: ignore
from dotenv import load_dotenv # type: ignore
load_dotenv()

#LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7
)

#States
class PipelineState(TypedDict):
    raw_input : str
    edited_text : str
    script_text : str
    final_output : str

#Nodes
def editor_node(state : PipelineState) -> dict:
    """Stage 1: Cleans up grammar, removes typos, and refines the tone."""
    print("\n--- [Stage 1] Executing the Editor Node ---")

    prompt = (
        "You are an expert copy editor. Your task is to clean up the following text by "
        "fixing all grammar, spelling, and punctuation errors, while refining the tone "
        "to be clear, engaging, and natural.\n\n"
        "Strict rules:\n"
        "- Preserve the original meaning and all key facts.\n"
        "- Do not add new information or change the core message.\n"
        "- Return ONLY the edited text. Do not include any introductory remarks, "
        "explanations, or markdown formatting around the output.\n\n"
        f"Text to edit:\n{state['raw_input']}"
    )

    response = llm.invoke(prompt)

    return {"edited_text" : response.content.strip()}

def scriptwriter_node(state : PipelineState) -> dict:
    """Stage 2: Formats the clean text into an engaging video script style"""
    print("\n--- [Stage 2] Executing the Scriptwriter Node ---")

    prompt = (
        "You are a professional video scriptwriter. Transform the following edited text "
        "into an engaging, spoken-word video script.\n\n"
        "Formatting & Style Guidelines:\n"
        "- Start with a strong, attention-grabbing hook.\n"
        "- Use short, punchy sentences written for natural spoken delivery.\n"
        "- Include brief visual or B-roll cues in brackets like [Visual: ...] where helpful.\n"
        "- End with a clear, concise takeaway or call to action.\n"
        "- Keep all facts and core ideas from the source text accurate.\n"
        "- Return ONLY the script. Do not include any introductory remarks, explanations, "
        "or closing commentary.\n\n"
        f"Source text:\n{state['edited_text']}"
    )

    response = llm.invoke(prompt)

    return {"script_text" : response.content.strip()}

def translator_node(state : PipelineState) -> dict:
    """Stage 3: Translate the script into natural flowing Hinglish"""
    print("\n--- [Stage 3] Executing the Hinglish Translator Node ---")

    prompt = (
        "You are an expert bilingual content creator. Translate the following video script "
        "into natural, conversational Hinglish (a smooth blend of everyday Hindi/Urdu and English "
        "written in the Roman/Latin alphabet).\n\n"
        "Strict Rules:\n"
        "- Write strictly in the Roman alphabet (do NOT use Devanagari or Urdu script).\n"
        "- Keep common English technical terms, verbs, and buzzwords in English so it sounds "
        "authentic and modern, not overly formal or textbook-translated.\n"
        "- Maintain the energetic, spoken-word rhythm and hook from the original script.\n"
        "- Keep all [Visual: ...] cues intact in English.\n"
        "- Return ONLY the translated Hinglish script. Do not include any introductory text, "
        "notes, or explanations.\n\n"
        f"Script to translate:\n{state['script_text']}"
    )

    response = llm.invoke(prompt)

    return {"final_output" : response.content.strip()}

#Graph
from langgraph.graph import StateGraph, START, END # type: ignore

# Define the state graph for the pipeline
graph = StateGraph(PipelineState)

# Add nodes to the graph
graph.add_node("editor", editor_node)
graph.add_node("scriptwriter", scriptwriter_node)
graph.add_node("translator", translator_node)   

#Add edges to define the flow between nodes
graph.add_edge(START, "editor")
graph.add_edge("editor", "scriptwriter")
graph.add_edge("scriptwriter", "translator")
graph.add_edge("translator", END)

#compile the graph
app = graph.compile()
result = app.invoke(
    {
        "raw_input" : "Ai agents are the future of the tech, They can think, plan and act their own"
            "LangGraph helps you build these agents with proper control and memory"
    }
    )

#output the final result
print("\n--- [Pipeline Execution Complete] Final Output ---")
print(f'Edited Text: {result["edited_text"]}')
print(f'Script Text: {result["script_text"]}')
print(f'Final Output: {result["final_output"]}')
