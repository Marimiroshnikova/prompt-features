"""Named word and phrase lists.

Every lexicon has a name, so when a feature fires the explanation can say
*which* list matched and *which* terms, instead of just showing a number.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Lexicon:
    name: str
    description: str
    terms: tuple[str, ...]

    @functools.cached_property
    def pattern(self) -> re.Pattern:
        # Longest first so "other than" wins over "than", and spaces tolerate
        # line breaks so a phrase split across lines still matches.
        parts = []
        for term in sorted(set(self.terms), key=len, reverse=True):
            body = r"\s+".join(re.escape(word) for word in term.split())
            prefix = r"\b" if term[:1].isalnum() else ""
            suffix = r"\b" if term[-1:].isalnum() else ""
            parts.append(f"{prefix}{body}{suffix}")
        return re.compile("|".join(parts), re.IGNORECASE)

    def find(self, text: str) -> list[dict]:
        return [
            {"start": m.start(), "end": m.end(), "text": m.group(0), "lexicon": self.name}
            for m in self.pattern.finditer(text)
        ]

    def hits(self, text: str) -> list[str]:
        seen: list[str] = []
        for match in self.find(text):
            token = match["text"].lower()
            if token not in seen:
                seen.append(token)
        return seen

    def matches(self, text: str) -> bool:
        return self.pattern.search(text) is not None


# --- logic and polarity -----------------------------------------------------

NEGATION = Lexicon(
    "negation",
    "Words that flip the polarity of the information need.",
    (
        "not", "never", "no", "none", "neither", "nor", "without", "cannot",
        "can't", "won't", "don't", "doesn't", "didn't", "isn't", "aren't",
        "wasn't", "weren't", "shouldn't", "wouldn't", "couldn't", "mustn't",
        "haven't", "hasn't", "hadn't", "ain't", "hardly", "barely", "scarcely",
        "nothing", "nobody", "nowhere", "lack of", "absence of", "fails to",
        "failed to",
        # "avoid", "prohibited" and "forbidden" are deliberately absent: they are
        # prohibition verbs, not negation, and counting them made
        # "What should be avoided?" look like a negated query.
    ),
)

EXCLUSION = Lexicon(
    "exclusion",
    "Phrases that carve an exception out of the main topic.",
    (
        "except", "excepting", "excluding", "exclude", "excludes", "other than",
        "besides", "apart from", "aside from", "rather than", "instead of",
        "but not", "not including", "leaving out", "omitting", "ignore",
        "ignoring", "skip", "disregard", "everything but", "all but",
    ),
)

CONDITIONAL = Lexicon(
    "conditional",
    "Markers of a hypothetical or conditional scope.",
    (
        "if", "unless", "provided that", "assuming", "assuming that", "in case",
        "whether", "suppose", "supposing", "given that", "only if", "as long as",
        "in the event", "otherwise", "hypothetically", "what if",
    ),
)

# --- reference and vagueness ------------------------------------------------

PRONOUNS = Lexicon(
    "pronouns",
    "Pronouns that need an antecedent to be resolvable.",
    (
        "it", "its", "it's", "they", "them", "their", "theirs", "this", "that",
        "these", "those", "he", "him", "his", "she", "her", "hers", "one",
        "ones", "such",
    ),
)

DANGLING_REFERENCE = Lexicon(
    "dangling_reference",
    "Pointers to context that was never included in the prompt.",
    (
        "the above", "as mentioned", "as mentioned above", "aforementioned",
        "the following", "as follows", "the below", "see below", "see above",
        "this document", "that document", "the document", "the attached",
        "attached file", "the file", "the text", "the passage", "the excerpt",
        "the article", "the paper", "the report", "the previous", "previously",
        "as discussed", "we discussed", "you mentioned", "as stated",
        "the same", "the former", "the latter", "earlier in", "in the snippet",
        "given context", "the context", "this code", "the code above",
        "my last message", "our conversation",
    ),
)

VAGUE_TERMS = Lexicon(
    "vague_terms",
    "Placeholder nouns that carry no retrievable content.",
    (
        "thing", "things", "stuff", "something", "someone", "somebody",
        "somewhere", "somehow", "anything", "anyone", "anywhere", "whatever",
        "whoever", "wherever", "etc", "et cetera", "and so on", "and so forth",
        "sort of", "kind of", "some kind of", "a bunch of", "and such",
        "or something", "the usual", "as usual", "general idea", "big picture",
        "miscellaneous", "misc",
    ),
)

VAGUE_QUANTIFIERS = Lexicon(
    "vague_quantifiers",
    "Quantifiers that do not pin down an amount.",
    (
        "some", "many", "few", "a few", "several", "most", "a lot", "lots of",
        "plenty", "numerous", "various", "multiple", "a couple", "a number of",
        "much", "little", "certain", "assorted",
    ),
)

HEDGES = Lexicon(
    "hedges",
    "Hedging language that loosens the requirement.",
    (
        "maybe", "perhaps", "possibly", "probably", "might", "may be",
        "could be", "seems", "seemingly", "apparently", "roughly",
        "approximately", "around", "about", "more or less", "i think",
        "i guess", "i believe", "not sure", "unsure", "somewhat", "fairly",
        "relatively", "arguably", "presumably", "tend to", "typically",
        "generally", "usually", "in theory", "ideally",
    ),
)

# --- time -------------------------------------------------------------------

TEMPORAL_RELATIVE = Lexicon(
    "temporal_relative",
    "Time references that depend on 'now', so they go stale in an index.",
    (
        "recent", "recently", "latest", "newest", "current", "currently",
        "now", "right now", "today", "nowadays", "these days", "modern",
        "present day", "at present", "up to date", "up-to-date", "as of",
        "so far", "to date", "lately", "just released", "upcoming", "soon",
        "this year", "this month", "this week", "last year", "last month",
        "last week", "next year", "next month", "next week", "yesterday",
        "tomorrow", "state of the art", "cutting edge", "still",
        "at the moment", "these years", "in the past year",
    ),
)

TEMPORAL_RANGE = Lexicon(
    "temporal_range",
    "Boundary words that restrict an answer to a period.",
    (
        "before", "after", "since", "until", "till", "between", "during",
        "prior to", "following", "up to", "as early as", "as late as",
        "earlier than", "later than", "no later than", "no earlier than",
        "throughout", "by the end of", "in the era of", "over the past",
        "in the last", "within the last", "ago",
    ),
)

MONTHS = Lexicon(
    "months",
    "Month names and common abbreviations.",
    (
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
        "oct", "nov", "dec",
    ),
)

WEEKDAYS = Lexicon(
    "weekdays",
    "Day names.",
    (
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
        "sunday",
    ),
)

# --- reasoning shape --------------------------------------------------------

COMPARISON = Lexicon(
    "comparison",
    "Phrases that require at least two sources to answer.",
    (
        "compare", "compared to", "compared with", "comparison", "versus",
        "vs", "vs.", "difference between", "differences between", "difference",
        "differences", "contrast", "in contrast", "better than", "worse than",
        "similarities", "similarity", "similar to", "pros and cons",
        "advantages and disadvantages", "tradeoff", "tradeoffs", "trade-off",
        "which is better", "which one", "rather than", "relative to",
        "against", "outperform", "prefer", "instead of", "both",
    ),
)

CAUSAL = Lexicon(
    "causal",
    "Cause-and-effect language, which is rarely stated in one chunk.",
    (
        "because", "cause", "causes", "caused", "causing", "reason", "reasons",
        "due to", "owing to", "leads to", "lead to", "led to", "results in",
        "resulted in", "effect of", "effects of", "impact of", "impacts of",
        "consequence", "consequences", "influence", "influences", "why",
        "explain why", "as a result", "therefore", "thus", "hence",
        "responsible for", "driven by", "stems from",
    ),
)

AGGREGATION = Lexicon(
    "aggregation",
    "Asks that need many records rolled up into one answer.",
    (
        "how many", "how much", "total", "sum", "sum of", "average", "mean",
        "median", "count", "number of", "most", "least", "highest", "lowest",
        "top", "bottom", "largest", "smallest", "biggest", "maximum", "minimum",
        "max", "min", "all of", "list all", "every", "overall",
        "combined", "aggregate", "percentage of", "share of", "ranking",
        # "best", "worst", "first" and "last" are deliberately absent: they are
        # far more often part of a name ("best picture", "First Lady") than a
        # genuine aggregation request.
    ),
)

SYNTHESIS = Lexicon(
    "synthesis",
    "Asks that require combining or reorganising several sources.",
    (
        "summarize", "summarise", "summary", "synthesize", "synthesise",
        "combine", "integrate", "consolidate", "overview", "across",
        "in general", "overall picture", "timeline", "history of",
        "evolution of", "trend", "trends", "review of", "survey of",
        "everything about", "comprehensive",
    ),
)

# --- task and category ------------------------------------------------------

TASK_VERBS: dict[str, str] = {
    "compare": "compare", "contrast": "compare", "summarize": "summarize",
    "summarise": "summarize", "explain": "explain", "describe": "describe",
    "list": "list", "enumerate": "list", "write": "write", "draft": "write",
    "compose": "write", "rewrite": "rewrite", "edit": "rewrite",
    "translate": "translate", "calculate": "calculate", "compute": "calculate",
    "solve": "calculate", "define": "define", "analyze": "analyze",
    "analyse": "analyze", "review": "review", "generate": "generate",
    "create": "generate", "make": "generate", "build": "build",
    "implement": "implement", "code": "implement", "debug": "debug",
    "fix": "fix", "optimize": "optimize", "refactor": "refactor",
    "find": "find", "search": "find", "identify": "identify",
    "evaluate": "evaluate", "assess": "evaluate", "classify": "classify",
    "categorize": "classify", "extract": "extract", "convert": "convert",
    "prove": "prove", "recommend": "recommend", "suggest": "recommend",
    "outline": "outline", "plan": "plan", "predict": "predict",
    "check": "check", "verify": "verify", "validate": "verify",
    "show": "show", "give": "give", "tell": "tell", "help": "help",
}

TASK_VERB_LEXICON = Lexicon(
    "task_verbs",
    "Imperative verbs naming the requested operation.",
    tuple(TASK_VERBS),
)

CODING = Lexicon(
    "coding",
    "Programming vocabulary.",
    (
        "def", "class", "import", "function", "python", "javascript",
        "typescript", "java", "c++", "rust", "golang", "sql", "algorithm",
        "code", "regex", "api", "endpoint", "compile", "compiler", "runtime",
        "variable", "array", "list comprehension", "dictionary", "loop",
        "recursion", "pointer", "async", "await", "promise", "callback",
        "stack trace", "traceback", "exception", "bug", "debug", "unit test",
        "pytest", "npm", "pip", "docker", "kubernetes", "git", "repository",
        "commit", "pull request", "framework", "library", "react", "django",
        "flask", "numpy", "pandas", "syntax error", "type error", "null",
        "undefined", "boolean", "integer", "string", "refactor", "linter",
    ),
)

CREATIVE = Lexicon(
    "creative",
    "Requests for invented rather than retrieved content.",
    (
        "story", "short story", "poem", "poetry", "creative", "fiction",
        "narrative", "write a song", "lyrics", "haiku", "sonnet", "novel",
        "screenplay", "script", "dialogue", "character", "plot", "fantasy",
        "sci-fi", "fairy tale", "joke", "limerick", "imagine a",
        "make up a", "invent a", "brainstorm",
    ),
)

FACT_RETRIEVAL = Lexicon(
    "fact_retrieval",
    "Signals of a single verifiable fact lookup.",
    (
        "who wrote", "who is", "who was", "who invented", "who founded",
        "who directed", "who created", "who discovered", "who won",
        "when did", "when was", "when is", "where is", "where was",
        "what year", "what date", "how tall", "how old", "how far",
        "how long is", "capital of", "population of", "born", "birthplace",
        "died", "founded", "invented", "discovered", "author of",
        "director of", "president of", "ceo of", "located", "located in",
        "headquarters", "according to wikipedia", "wikipedia article",
        "claim and evidence", "fact verification", "fact check", "is it true",
        "release date", "released in", "name of",
    ),
)

SUMMARIZATION = Lexicon(
    "summarization",
    "Requests to compress a source.",
    (
        "summarize", "summarise", "summary", "tl;dr", "tldr", "key points",
        "main points", "key takeaways", "takeaways", "overview", "abstract",
        "gist", "condense", "recap", "in brief", "bullet summary",
        "high level", "executive summary",
    ),
)

DEFINITION = Lexicon(
    "definition",
    "Requests for the meaning of a term.",
    (
        "what is", "what are", "what does", "define", "definition",
        "meaning of", "means", "stand for", "stands for", "terminology",
        "concept of", "in simple terms", "explain the concept", "what's a",
        "difference between a",
    ),
)

MATH = Lexicon(
    "math",
    "Arithmetic and formal maths vocabulary.",
    (
        "calculate", "compute", "solve", "equation", "formula", "derivative",
        "integral", "probability", "percentage", "percent of", "sum of",
        "multiply", "divide", "subtract", "add up", "square root", "algebra",
        "geometry", "matrix", "vector", "theorem", "proof", "logarithm",
        "factorial", "arithmetic", "how much is", "what is the value of",
        "round to", "decimal", "fraction", "ratio", "standard deviation",
        "variance", "mean of",
    ),
)

TRANSLATION = Lexicon(
    "translation",
    "Cross-language requests.",
    (
        "translate", "translation", "in spanish", "in french", "in german",
        "in italian", "in portuguese", "in russian", "in chinese",
        "in japanese", "in korean", "in arabic", "in hindi", "into english",
        "from english", "to english", "how do you say", "say in",
        "what does it mean in",
    ),
)

# --- domains ----------------------------------------------------------------

DOMAIN_LEXICONS: tuple[Lexicon, ...] = (
    Lexicon(
        "domain_medical",
        "Clinical and pharmacological vocabulary.",
        (
            "patient", "patients", "dose", "dosage", "mg", "symptom",
            "symptoms", "diagnosis", "diagnose", "treatment", "therapy",
            "drug", "drugs", "medication", "ibuprofen", "aspirin",
            "acetaminophen", "paracetamol", "antibiotic", "antibiotics",
            "nsaid", "nsaids", "vaccine", "clinical trial", "side effect",
            "side effects", "contraindication", "fever", "infection",
            "cancer", "diabetes", "blood pressure", "cardiac", "surgery",
            "physician", "doctor", "nurse", "icu", "mortality", "pediatric",
            "dementia", "insulin", "chemotherapy", "prognosis", "syndrome",
        ),
    ),
    Lexicon(
        "domain_legal",
        "Law and regulation vocabulary.",
        (
            "law", "laws", "legal", "statute", "regulation", "regulatory",
            "contract", "clause", "liability", "plaintiff", "defendant",
            "court", "supreme court", "lawsuit", "litigation", "attorney",
            "lawyer", "jurisdiction", "compliance", "gdpr", "hipaa",
            "patent", "trademark", "copyright", "tort", "felony",
            "misdemeanor", "indemnity", "arbitration", "subpoena", "verdict",
            "appeal", "constitutional", "amendment", "treaty",
        ),
    ),
    Lexicon(
        "domain_finance",
        "Money, markets and accounting vocabulary.",
        (
            "revenue", "profit", "loss", "ebitda", "margin", "earnings",
            "stock", "stocks", "share price", "shares", "dividend", "bond",
            "bonds", "interest rate", "inflation", "gdp", "market cap",
            "valuation", "portfolio", "investment", "investor", "hedge fund",
            "balance sheet", "cash flow", "income statement", "10-k", "10-q",
            "filing", "quarterly", "fiscal year", "tax", "taxes", "audit",
            "budget", "cost", "pricing", "roi", "capital", "debt", "equity",
            "crypto", "bitcoin", "ethereum",
        ),
    ),
    Lexicon(
        "domain_tech",
        "Software, hardware and IT vocabulary.",
        (
            "software", "hardware", "server", "database", "cloud", "aws",
            "azure", "gcp", "latency", "throughput", "bandwidth", "cpu",
            "gpu", "ram", "storage", "network", "protocol", "http", "https",
            "tcp", "dns", "encryption", "authentication", "authorization",
            "oauth", "token", "cache", "queue", "microservice", "container",
            "deployment", "ci/cd", "pipeline", "machine learning",
            "neural network", "model", "training", "embedding", "llm",
            "transformer", "dataset", "vector database", "rag", "retrieval",
            "operating system", "linux", "windows", "macos", "browser",
        ),
    ),
    Lexicon(
        "domain_science",
        "Natural-science vocabulary.",
        (
            "experiment", "hypothesis", "theory", "physics", "chemistry",
            "biology", "molecule", "atom", "electron", "proton", "neutron",
            "quantum", "relativity", "gravity", "energy", "entropy",
            "thermodynamics", "photosynthesis", "dna", "rna", "gene",
            "genome", "protein", "enzyme", "cell", "species", "evolution",
            "ecosystem", "climate", "carbon", "emissions", "temperature",
            "telescope", "galaxy", "planet", "orbit", "asteroid", "nasa",
            "peer reviewed", "journal", "citation",
        ),
    ),
    Lexicon(
        "domain_history",
        "Historical and geopolitical vocabulary.",
        (
            "war", "world war", "revolution", "empire", "dynasty", "kingdom",
            "colonial", "colonization", "treaty", "battle", "ancient",
            "medieval", "renaissance", "century", "bc", "ad", "civilization",
            "pharaoh", "emperor", "king", "queen", "president", "election",
            "independence", "constitution", "cold war", "communism",
            "democracy", "monarchy", "invasion", "conquest", "archaeology",
        ),
    ),
    Lexicon(
        "domain_business",
        "Company and operations vocabulary.",
        (
            "company", "startup", "customer", "customers", "client",
            "marketing", "sales", "strategy", "competitor", "competitors",
            "product", "launch", "roadmap", "stakeholder", "kpi", "okr",
            "churn", "retention", "conversion", "funnel", "b2b", "b2c",
            "saas", "supply chain", "logistics", "vendor", "procurement",
            "hr", "hiring", "onboarding", "employee", "manager", "ceo",
            "board", "merger", "acquisition", "partnership",
        ),
    ),
    Lexicon(
        "domain_everyday",
        "Consumer, travel, food and lifestyle vocabulary.",
        (
            "recipe", "cook", "cooking", "bake", "ingredient", "restaurant",
            "flight", "hotel", "travel", "visa", "passport", "itinerary",
            "weather", "workout", "exercise", "diet", "calories", "sleep",
            "movie", "film", "song", "album", "band", "game", "sport",
            "football", "soccer", "basketball", "tennis", "car", "phone",
            "laptop", "shopping", "price of", "review of", "how to clean",
            "how to fix",
        ),
    ),
)

# --- prompt scaffolding -----------------------------------------------------

ROLE_PROMPT = Lexicon(
    "role_prompt",
    "Persona assignment aimed at the model, not at the retriever.",
    (
        "you are a", "you are an", "you are the", "you're a", "you're an",
        "act as", "acting as", "pretend to be", "pretend you are",
        "imagine you are", "imagine that you are", "as an expert",
        "your role is", "your job is", "your task is", "behave like",
        "respond as", "you will act", "assume the role", "take the role",
        "you are chatgpt", "you are an ai", "you are a helpful",
    ),
)

FORMAT_REQUEST: dict[str, str] = {
    "json": "json", "in json": "json", "json format": "json",
    "xml": "xml", "yaml": "yaml", "csv": "csv", "markdown": "markdown",
    "table": "table", "in a table": "table", "tabular": "table",
    "bullet points": "bullets", "bulleted list": "bullets",
    "bullet list": "bullets", "bullets": "bullets",
    "numbered list": "numbered_list", "ordered list": "numbered_list",
    "code block": "code", "code snippet": "code", "pseudocode": "code",
    "one sentence": "one_sentence", "a single sentence": "one_sentence",
    "one paragraph": "one_paragraph", "short answer": "short_answer",
    "one word": "one_word", "yes or no": "yes_no", "tl;dr": "tldr",
    "essay": "essay", "step by step": "steps", "step-by-step": "steps",
    "numbered steps": "steps", "outline": "outline", "diagram": "diagram",
    "flowchart": "diagram", "checklist": "checklist", "report": "report",
    "email": "email", "letter": "letter", "headline": "headline",
}

FORMAT_LEXICON = Lexicon(
    "output_format",
    "Requests for a particular response shape.",
    tuple(FORMAT_REQUEST),
)

LENGTH_LIMIT = Lexicon(
    "length_limit",
    "Explicit limits on answer length.",
    (
        "no more than", "at most", "at least", "fewer than", "less than",
        "under", "within", "maximum of", "up to", "briefly", "in brief",
        "be concise", "concise", "short", "keep it short", "in a few words",
        "in one line", "word limit", "character limit", "words or less",
        "words or fewer", "sentences or less", "limit your",
    ),
)

CITATION_REQUEST = Lexicon(
    "citation_request",
    "Demands for sources, which raise the retrieval bar.",
    (
        "cite", "cite your", "citation", "citations", "source", "sources",
        "reference", "references", "footnote", "footnotes", "with links",
        "provide evidence", "back it up", "according to", "quote the",
        "page number", "bibliography", "attribution", "where did you find",
    ),
)

COT_CUE = Lexicon(
    "chain_of_thought",
    "Instructions to reason out loud.",
    (
        "step by step", "step-by-step", "think through", "think carefully",
        "reasoning", "chain of thought", "show your work",
        "show your reasoning", "let's think", "let us think",
        "explain your reasoning", "walk me through", "break it down",
        "work through", "before answering", "first think",
    ),
)

POLITENESS = Lexicon(
    "politeness",
    "Filler courtesy that adds tokens but no retrievable content.",
    (
        "please", "thanks", "thank you", "could you", "would you",
        "can you", "kindly", "i'd like", "i would like", "if possible",
        "appreciate", "sorry", "excuse me", "help me", "i need you to",
        "for me", "hi", "hello", "hey",
    ),
)

META_INSTRUCTION = Lexicon(
    "meta_instruction",
    "Sentences that tell the model how to answer rather than what to find.",
    (
        "use only", "use the docs", "use the documents", "use the context",
        "using only", "based on the", "based only on", "answer in",
        "answer using", "answer only", "answer the question",
        "answer with", "respond in", "respond with", "respond only",
        "reply in", "reply with", "do not", "don't include", "never include",
        "output", "output only", "return only", "format your", "in the format",
        "cite your", "cite sources", "provide sources", "be concise",
        "be brief", "be specific", "keep it", "make sure", "ensure that",
        "please note", "remember to", "you must", "you should", "your task",
        "your goal", "your job", "you are", "act as", "as an expert",
        "think step", "step by step", "let's think", "avoid using",
        "if you don't know", "if unsure", "no preamble", "no explanation",
        "follow these", "follow the", "adhere to", "stick to",
        "helpful assistant", "you will be given", "given the following",
        "read the following", "consider the following",
    ),
)

ENUMERATION_REQUEST = Lexicon(
    "enumeration_request",
    "Asks for a set of items rather than one answer.",
    (
        "list", "list of", "give me", "name a few", "name some", "examples of",
        "example of", "enumerate", "top", "best", "several", "options",
        "alternatives", "ways to", "types of", "kinds of", "reasons why",
        "ideas for", "suggestions", "bullets", "bullet points", "bulleted",
        "each of", "all of the", "pros and cons", "advantages and disadvantages",
    ),
)

# --- fallback stopwords (used when spaCy is unavailable) --------------------

STOPWORDS: frozenset[str] = frozenset(
    """
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for from
further had hadn't has hasn't have haven't having he her here hers herself him
himself his how i i'd i'll i'm i've if in into is isn't it it's its itself
let's me more most mustn't my myself no nor not of off on once only or other
ought our ours ourselves out over own same shan't she should shouldn't so some
such than that the their theirs them themselves then there these they this
those through to too under until up very was wasn't we were weren't what when
where which while who whom why with won't would wouldn't you your yours
yourself yourselves will just also may might must shall does did done
""".split()
)

NUMBER_WORDS: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
    "thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000,
    "dozen": 12, "couple": 2, "first": 1, "second": 2, "third": 3,
    "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
    "ninth": 9, "tenth": 10,
}

NUMBER_WORD_LEXICON = Lexicon(
    "number_words",
    "Numbers written as words.",
    tuple(NUMBER_WORDS),
)

UNITS = Lexicon(
    "units",
    "Measurement units that make a fact checkable.",
    (
        "mg", "mcg", "kg", "gram", "grams", "lb", "lbs", "pound", "pounds",
        "oz", "ounce", "ounces", "km", "kilometre", "kilometer", "meter",
        "metre", "meters", "cm", "mm", "mile", "miles", "foot", "feet",
        "inch", "inches", "gb", "mb", "tb", "kb", "byte", "bytes", "hz",
        "khz", "mhz", "ghz", "ml", "litre", "liter", "liters", "gallon",
        "celsius", "fahrenheit", "kelvin", "degrees", "bpm", "mph", "kph",
        "watt", "watts", "volt", "volts", "amp", "amps", "joule", "calorie",
        "calories", "second", "seconds", "minute", "minutes", "hour",
        "hours", "day", "days", "week", "weeks", "month", "months", "year",
        "years", "percent", "percentage points", "pixels", "px",
    ),
)

CURRENCY = Lexicon(
    "currency",
    "Currency symbols and codes.",
    (
        "$", "€", "£", "¥", "₹", "usd", "eur", "gbp", "jpy", "cny", "inr",
        "chf", "cad", "aud", "dollar", "dollars", "euro", "euros", "pound",
        "pounds sterling", "yen", "rupee", "rupees", "cents",
    ),
)

ALL_LEXICONS: dict[str, Lexicon] = {
    lex.name: lex
    for lex in (
        NEGATION, EXCLUSION, CONDITIONAL, PRONOUNS, DANGLING_REFERENCE,
        VAGUE_TERMS, VAGUE_QUANTIFIERS, HEDGES, TEMPORAL_RELATIVE,
        TEMPORAL_RANGE, MONTHS, WEEKDAYS, COMPARISON, CAUSAL, AGGREGATION,
        SYNTHESIS, TASK_VERB_LEXICON, CODING, CREATIVE, FACT_RETRIEVAL,
        SUMMARIZATION, DEFINITION, MATH, TRANSLATION, ROLE_PROMPT,
        FORMAT_LEXICON, LENGTH_LIMIT, CITATION_REQUEST, COT_CUE, POLITENESS,
        META_INSTRUCTION, ENUMERATION_REQUEST, NUMBER_WORD_LEXICON, UNITS,
        CURRENCY,
        *DOMAIN_LEXICONS,
    )
}
