"""Compare models as judges of long answers: which marking points does the answer make?

One real answer (Class 10 SST booklet, Q25, 4 paragraphs, transcribed by hand) and 10 single-idea
marking points: 6 the student made and 4 decoys that reuse the answer's words without its meaning.
Each point is asked about separately; the model must quote the student's words, and a "yes" counts
only when that quote is really in the answer (points.evidence_found).

    python eval/bench_judge.py MODEL [MODEL ...]
"""
import json
import re
import sys
import time
import urllib.request

from model import OLLAMA_URL
from points import evidence_found

QUESTION = "Explain how folklore, symbols and icons helped create a sense of collective belonging among Indians."
ANSWER = (
    "Indian folklore and symbols strengthened the idea of nationalism during the 20th century in the following ways - "
    "Collecting and recording folk culture was instrumental to the process of nation building as it reflected the "
    "national literature and was the most trustworthy account of people's manifestations and thoughts in the country. "
    "Rabindranath Tagore himself collected myths, rhyms (nursery rhymes), folk tales etc in Bengal while Natesa Sastri "
    "published a massive four volume book named 'Folklore of Southern India'. Villages were toured extensively to "
    "collect songs, legends, tales sung by the ballards.\n"
    "2. Allegories were prevelant during this period which provided a concrete form to the abstract idea of freedom. "
    "Indians visually associated the identity of India with the image of Bharat Mata which was first created by "
    "Bankim Chandra Chattopadhya and later on, Abanindranath Tagore made an ascetic, calm, composed image of Bharat "
    "Mata, dispensing learning and food along with clothing. Mr. Chattopadhya also composed a hymn called "
    "'Vande Matram' which was widely sung during the Swadeshi movement in Bengal.\n"
    "3. A tri-colour flag made up of yellow, red and green colours, comprising eight lotuses representing british "
    "provinces along with a crescent moon symbolising muslims & hindus was designed during the Swadeshi movement. "
    "Gandhi ji later on designed his swaraj flag made up of red, white and green colours with a 'charkha' in the "
    "middle symbolising Gandhi's idea of self help. It was a symbol of defiance against the british authorities.\n"
    "Conclusion - All these processes / activities along with reinterpretation of history were a part of wider "
    "cultural movement to instill a sense of collective belonging amongst the indians. History was reinterpreted "
    "and taught differently to rediscover one's identity and take pride in our victorious achievements in "
    "different fields like science, maths, law etc.")
POINTS = [  # (point, made by the student?)
    ("Folk tales, songs and legends were collected (e.g. by Rabindranath Tagore or Natesa Sastri).", True),
    ("Bharat Mata became the image of India (Bankim Chandra Chattopadhyay, Abanindranath Tagore).", True),
    ("The hymn Vande Mataram was sung during the Swadeshi movement.", True),
    ("A tricolour flag with eight lotuses and a crescent moon was designed in the Swadeshi movement.", True),
    ("Gandhiji's Swaraj flag had a spinning wheel (charkha) at the centre.", True),
    ("History was reinterpreted so that Indians could take pride in their past achievements.", True),
    ("Raja Ravi Varma painted images of Indian heroes and gods that were sold across India.", False),
    ("Abanindranath Tagore's paintings were sold in British markets to raise funds.", False),
    ("The Jallianwala Bagh massacre made people angry and united them.", False),
    ("Folk theatre such as Jatra was performed in villages to spread nationalist messages.", False),
]
PROMPT = """You are helping a teacher mark a student's exam answer.

Question: {question}

Student's answer:
\"\"\"
{answer}
\"\"\"

Marking point: {point}

Does the student's answer clearly make this point? Different wording is fine, but the idea itself must be
in the answer; words that merely appear in it are not enough. If it does, copy the student's exact words
that make the point (one sentence or phrase) into "quote". If not, set "made" to false and "quote" to "".
"""
SCHEMA = {"type": "object", "properties": {"made": {"type": "boolean"}, "quote": {"type": "string"}},
          "required": ["made", "quote"]}


def judge(model: str, point: str) -> tuple[bool, bool, str]:
    """(model said yes, yes with a quote really in the answer, quote)"""
    body = {"model": model, "stream": False, "format": SCHEMA,
            "messages": [{"role": "user", "content": PROMPT.format(question=QUESTION, answer=ANSWER, point=point)}],
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 4000}}
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        content = json.load(resp)["message"]["content"]
    try:
        out = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.S)
        out = json.loads(m.group(0)) if m else {"made": False, "quote": ""}
    said = bool(out.get("made"))
    quote = str(out.get("quote", ""))
    return said, said and evidence_found(quote, ANSWER), quote


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    for model in sys.argv[1:]:
        start = time.time()
        right = false_credit = missed = 0
        for point, made in POINTS:
            said, counted, quote = judge(model, point)
            ok = counted == made
            right += ok
            false_credit += counted and not made
            missed += made and not counted
            mark = "ok " if ok else "BAD"
            print(f"  {mark} {'made' if made else 'decoy':5} said={'yes' if said else 'no ':3} "
                  f"counted={'yes' if counted else 'no ':3} | {point[:55]} | {quote[:60]}", flush=True)
        print(f"{model}: {right}/10 right, false credit {false_credit}, missed {missed}, "
              f"{(time.time() - start) / len(POINTS):.0f} s/point\n", flush=True)


if __name__ == "__main__":
    main()
