
import streamlit as st
from transformers import pipeline

@st.cache_resource  # Load the model once, not on every click
def load_model():
    return pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
    )

classifier = load_model()

# ==========================================================
# PASTE FROM COLAB: START
# Paste the three cells marked "COPY THIS" in Colab:
# ROUTES + choice(), noul(), URGENCY + score().
# ==========================================================

ROUTES = ["billing", "technical problem", "sales question", "thank you"]

def choice(text, options, template="This message is about {}."):
    r = classifier(
        text,
        candidate_labels=options,
        hypothesis_template=template
    )
    return {
        label: round(p, 3)
        for label, p in zip(r["labels"], r["scores"])
    }

def noul(text, yes, no, template="The customer is {}."):
    return choice(text, [yes, no], template)[yes]

URGENCY = [
    "it can wait a week",
    "it should be handled today",
    "it needs action right now"
]

def score(text, levels, template="For support, {}."):
    probs = choice(text, levels, template)
    value = sum((levels.index(level) + 1) * p for level, p in probs.items())
    return round(value, 2)

# ==========================================================
# PASTE FROM COLAB: END
# ==========================================================
