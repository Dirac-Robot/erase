"""
NIAH Test - English Wikipedia Document (Binary Logic)

Uses binary is_relevant/should_exclude flags.
"""
import asyncio
import re
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from ato.adict import ADict

from erase import scope


BATCH_SIZE = 30

# English Wikipedia-style content
DOCUMENT = """
Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.

Guido van Rossum began working on Python in the late 1980s as a successor to the ABC programming language and first released it in 1991 as Python 0.9.0. Python 2.0 was released in 2000. Python 3.0, released in 2008, was a major revision not completely backward-compatible with earlier versions. The transition from Python 2 to Python 3 took many years.

Python consistently ranks as one of the most popular programming languages. It is used for web development, data science, artificial intelligence, and scientific computing. Many companies rely on Python for their backend systems.

The Python Software Foundation manages and directs resources for Python development. The foundation was founded in 2001 by Guido van Rossum. The PSF organizes PyCon, the largest Python conference.

Django is a popular web framework written in Python. It was created by Adrian Holovaty and Simon Willison in 2003. Flask is another popular micro web framework for Python, created by Armin Ronacher in 2010. FastAPI is a newer framework created by Sebastian Ramirez in 2018.

NumPy is a library for scientific computing in Python. It was created by Travis Oliphant in 2005. Pandas is a data manipulation library built on top of NumPy, created by Wes McKinney in 2008. SciPy extends NumPy with additional scientific algorithms.

TensorFlow is an open-source machine learning library developed by Google. It was released in 2015. PyTorch is a competing library developed by Facebook, released in 2016. Both frameworks support GPU acceleration for training neural networks.

The Zen of Python is a collection of 19 guiding principles for writing computer programs in Python. It was written by Tim Peters in 1999 and can be accessed by typing 'import this' in Python. The principles include "Beautiful is better than ugly" and "Simple is better than complex".

Python's creator Guido van Rossum was born in the Netherlands in 1956. He worked at Google from 2005 to 2012, then at Dropbox from 2013 to 2019. He is known as Python's "Benevolent Dictator For Life" (BDFL). He stepped down from this role in 2018.

JavaScript is another popular programming language, primarily used for web development. It was created by Brendan Eich in 1995. Node.js allows JavaScript to run on servers.

Java is a statically typed language developed by Sun Microsystems. It was first released in 1995. Java runs on the Java Virtual Machine (JVM). Many enterprise applications use Java.

Rust is a systems programming language created by Mozilla. It was first released in 2010. Rust focuses on memory safety without garbage collection. The Linux kernel now includes Rust code.

Go was created at Google by Robert Griesemer, Rob Pike, and Ken Thompson. It was released in 2009. Go is known for its simplicity and built-in concurrency support.

Ruby is a dynamic language created by Yukihiro Matsumoto in 1995. Ruby on Rails is a popular web framework. Many startups used Ruby for rapid development.

Kotlin is a modern language that runs on the JVM. It was created by JetBrains and released in 2011. Google announced Kotlin as an official Android language in 2017.

Swift is Apple's programming language for iOS and macOS development. It was released in 2014. Swift replaced Objective-C for Apple platform development.

TypeScript is a superset of JavaScript created by Microsoft. It was released in 2012. TypeScript adds static typing to JavaScript.

Machine learning has transformed many industries. Deep learning uses neural networks with many layers. Natural language processing enables computers to understand text. Computer vision allows machines to interpret images.

Data science combines statistics, programming, and domain knowledge. Data scientists often use Python and R. Jupyter notebooks are popular for data analysis. Kaggle hosts data science competitions.

Cloud computing provides on-demand computing resources. AWS, Azure, and Google Cloud are major providers. Serverless computing abstracts away server management. Containers like Docker simplify deployment.

DevOps combines development and operations practices. CI/CD pipelines automate software delivery. Kubernetes orchestrates container deployments. Infrastructure as code manages cloud resources.

Cybersecurity protects systems from attacks. Encryption secures data in transit and at rest. Firewalls block unauthorized access. Security testing identifies vulnerabilities.

Blockchain is a distributed ledger technology. Bitcoin was the first cryptocurrency. Ethereum enables smart contracts. NFTs use blockchain for digital ownership.

Quantum computing uses quantum mechanical phenomena. IBM and Google develop quantum computers. Quantum algorithms could break current encryption.

The history of computing began with mechanical calculators. ENIAC was one of the first electronic computers. The transistor revolutionized electronics. Moore's Law predicted chip density doubling.

Open source software is freely available. Linux is a popular open source operating system. GitHub hosts millions of open source projects. The GPL and MIT are common licenses.
"""


class SentenceScore(BaseModel):
    id: str = Field(description="Unique ID (s1, s2, etc)")
    content: str = Field(description="The sentence content")
    is_relevant: bool = Field(description="Is this related to the query?")
    should_exclude: bool = Field(description="Should this be EXCLUDED? True=related but doesn't answer")


class BatchScoreResult(BaseModel):
    chunks: list[SentenceScore]


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def create_erase_prompt(sentences_text: str, query: str) -> str:
    """Simplified binary ERASE prompt with clear examples."""
    return f"""You are a document filter. For each sentence, decide:

### is_relevant
- true: Related to the query topic
- false: Completely unrelated

### should_exclude
- false (KEEP): Sentence doesn't have to be excluded.
- true (EXCLUDE): Sentence has to be excluded.

**QUERY: "{query}"**

## EXAMPLES
Query: "Who developed Linux?"
- "Linux was developed by Linus Torvalds in 1991." → is_relevant=true, should_exclude=false (KEEP: has the answer "Linus Torvalds")
- "Linux is open-source and free to use." → is_relevant=true, should_exclude=true (EXCLUDE: about Linux, but no answer to "Who")
- "Python is a programming language." → is_relevant=false, should_exclude=true (EXCLUDE: unrelated)

## SENTENCES:
{sentences_text}"""


def create_single_prompt(sentences_text: str, query: str) -> str:
    return f"""Identify sentences relevant to the query.

**QUERY: "{query}"**

## SENTENCES:
{sentences_text}

- is_relevant: true if related, false if not
- should_exclude: Always false"""


class BatchScorer:
    def __init__(self, config: ADict, use_erasure: bool):
        self._llm = ChatOpenAI(model=config.model)
        self._use_erasure = use_erasure
    
    def filter_batch(self, sentences: list[str], query: str) -> list[str]:
        structured_llm = self._llm.with_structured_output(BatchScoreResult)
        sentences_text = '\n'.join(f"[{i+1}] {s}" for i, s in enumerate(sentences))
        
        prompt = create_erase_prompt(sentences_text, query) if self._use_erasure else create_single_prompt(sentences_text, query)
        result: BatchScoreResult = structured_llm.invoke(prompt)
        
        # Original content map
        id_to_original = {f"s{i+1}": s for i, s in enumerate(sentences)}
        
        if self._use_erasure:
            kept_ids = [s.id for s in result.chunks if s.is_relevant and not s.should_exclude]
        else:
            kept_ids = [s.id for s in result.chunks if s.is_relevant]
            
        return [id_to_original[cid] for cid in kept_ids if cid in id_to_original]


async def score_all(scorer, sentences: list[str], query: str) -> list[str]:
    batches = [sentences[i:i+BATCH_SIZE] for i in range(0, len(sentences), BATCH_SIZE)]
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, scorer.filter_batch, batch, query) for batch in batches]
    results = await asyncio.gather(*tasks)
    return [s for batch in results for s in batch]


def answer(llm, context: list[str], query: str) -> str:
    context_text = '\n'.join(context[:10])
    prompt = f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
    return llm.invoke(prompt).content


@scope
def main(config):
    erase_scorer = BatchScorer(config, use_erasure=True)
    single_scorer = BatchScorer(config, use_erasure=False)
    llm = ChatOpenAI(model=config.model)
    
    sentences = split_sentences(DOCUMENT)
    
    tests = [
        {"q": "Who created Python?", "a": "Guido van Rossum"},
        {"q": "When was PyTorch released?", "a": "2016"},
        {"q": "Who wrote the Zen of Python?", "a": "Tim Peters"},
    ]
    
    async def run():
        for t in tests:
            q, expected = t['q'], t['a']
            print(f"Q: {q}")
            
            single_kept = await score_all(single_scorer, sentences, q)
            erase_kept = await score_all(erase_scorer, sentences, q)
            
            print(f"  Single: {len(single_kept)} kept | ERASE: {len(erase_kept)} kept")
            
            s_ans = answer(llm, single_kept, q)
            e_ans = answer(llm, erase_kept, q)
            
            print(f"  Single Answer: {s_ans[:50]}... {'✅' if expected.lower() in s_ans.lower() else '❌'}")
            print(f"  ERASE Answer:  {e_ans[:50]}... {'✅' if expected.lower() in e_ans.lower() else '❌'}")
            
            if len(single_kept) > 0:
                print(f"  Reduction: {1-len(erase_kept)/len(single_kept):.0%}")
            print()
    
    asyncio.run(run())


if __name__ == "__main__":
    main()
