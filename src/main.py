import sys
import os
import time

print(f"CWD: {os.getcwd()}")
os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers"
if os.path.exists("venv"):
    curr_dir = os.getcwd()
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    sys.path.append(f"{curr_dir}/venv/lib/python{version}/site-packages")
    print(f"New Path: {sys.path}")

print(f"Model Files Before: {os.listdir('./transformers/qamodel')}")

from transformers import pipeline
# question_answerer = pipeline("question-answering", model='distilbert-base-uncased-distilled-squad')
question_answerer = pipeline("question-answering", model='./transformers/qamodel')
print(f"Model Files After: {os.listdir('./transformers/qamodel')}")
# print(f"Cache Files After: {os.listdir(os.environ['TRANSFORMERS_CACHE'])}")


def main(*args):
    args = args[0]
    print(f"Args: {args}")
    model_input = [{"context": context, "question": question} for (context, question) in args]
    print(f"Model Input: {model_input}")
    t1 = time.time()
    answers = question_answerer(model_input)
    t2 = time.time()
    print(f"Infer Time: {(t2 - t1) * 1000}ms")
    print(f"Answers: {answers}")
    if not isinstance(answers, list):
        # Special case with one input.
        answers = [answers]
    answers = [answer["answer"] for answer in answers]
    print(f"Answers: {answers}")
    return answers
    
if __name__ == "__main__":
    result = question_answerer([
    {
        "question": "Who is boss?",
        "context": "Amadou is boss!"
    },
    {
        "question": "Who is boss?",
        "context": "Amadou is boss!"
    }
    ])
    print("AAAAA")
    print(result)
    print("AAAAA")

    result = question_answerer(question="Who is the boss?", context="Amadou is the boss")
    print(result["answer"])
