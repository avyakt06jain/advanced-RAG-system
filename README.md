
Solution Architecture: An Advanced Retrieval-Augmented Generation System for Insurance Policy Analysis


I. Deconstructing the Challenge: The Unique Complexities of Insurance Policy Documents


Introduction to the Problem Domain

The challenge presented by Hackrx 6.0 transcends a conventional text extraction or document summarization task. It ventures into a high-stakes domain where the accurate and reliable interpretation of complex legal and financial documents is paramount. An insurance policy is not merely a collection of text; it is a binding contract whose misinterpretation can have significant financial and personal consequences for the policyholder. Therefore, the objective is not simply to build a system that can "read" a PDF, but to engineer an intelligent information retrieval system that can answer nuanced, specific questions with verifiable accuracy and complete explainability. The system must function as a trusted expert, capable of navigating dense jargon, complex logical structures, and distributed information to provide answers that are grounded in the specific text of the policy document. This necessitates an architecture that prioritizes factual correctness, mitigates the risk of AI "hallucinations," and provides clear, traceable citations for every piece of generated information.

Deep Dive into Sample Documents

A thorough analysis of the provided sample documents—Edelweiss's "Well Baby Well Mother" add-on 1, Cholamandalam's "Group Domestic Travel Insurance" policy 1, and Bajaj Allianz's "Global Health Care" policy 1—reveals the multifaceted nature of the challenge. These documents are representative of the complex artifacts that the system must be designed to handle.

Structural Complexity

The structural layout of these documents presents a significant initial hurdle for any automated system. The Cholamandalam policy 1, a comprehensive 101-page document, exemplifies this complexity. It employs multi-column layouts, which can cause naive text extraction tools to read lines across columns, resulting in nonsensical, jumbled text. Furthermore, critical information regarding coverages and, more importantly, exclusions is often presented in deeply nested, multi-level lists.1 Preserving the hierarchical relationship of these nested points is essential to understanding their logical scope. Headers and footers containing document identifiers (like UINs and company details) are present on every page and can contaminate the main body of extracted text if not properly identified and removed. The Bajaj Allianz policy 1 introduces another layer of structural complexity with embedded tables of benefits that mix textual descriptions with numerical limits, requiring a parser that can understand both content types within a single structure.

Semantic Complexity

The language used in insurance policies is intentionally precise, dense, and laden with domain-specific terminology that carries specific legal weight. Terms such as "Subrogation," "Condition Precedent," "Indemnity," and "Deductible" are not common in everyday language but are fundamental to the contract's meaning.1 A general-purpose language model, trained on broad internet text, may lack the nuanced understanding required to interpret these terms correctly in context. Sentences are often long and grammatically complex, featuring extensive cross-referencing that is critical to the policy's logic. For example, a clause might state that a benefit is "subject to the terms, conditions and exclusions herein contained or otherwise expressed" 1, which requires the system to understand that the information in one section is conditional upon information located elsewhere in the document. This distributed nature of policy logic is a core semantic challenge.

Data in Tabular Format

A critical and often underestimated challenge is the extraction and interpretation of tabular data. The "Table of Benefits" in the Cholamandalam policy, detailing compensation percentages for various forms of disablement, is a prime example.1 Similarly, the benefit tables in the Bajaj Allianz policy present a structured relationship between coverages, plans ("Imperial Plan" vs. "Imperial Plus Plan"), and monetary limits in both Indian Rupees and US Dollars.1 A simple text extraction that linearizes this data into a flat string would destroy its relational integrity. The system must recognize that "100%" is the compensation percentage specifically for "Accidental Death (AD)" and not for "Loss of toes - all," which is 20%.1 This requires a specialized approach to identify, parse, and represent tabular data in a structured format that an AI model can correctly interpret.

Implicit Information Hierarchy

The logic of an insurance policy is inherently hierarchical and distributed. A specific coverage, such as "Emergency Accidental Hospitalisation," is introduced as a "Base Cover".1 However, the actual extent of this coverage is not defined in that section alone. It is invariably modified by clauses in the "General Exclusions" section, which might preclude claims related to pre-existing conditions or specific activities.1 It may be further refined by "Special Conditions" applicable to that specific type of cover. To answer a question like, "Am I covered for hospitalization if I get injured while rock climbing?", the system cannot simply look at the hospitalization section. It must synthesize information from the base coverage, the exclusions related to "Adventure Sports" 1, and any special conditions to form a complete and accurate answer. This requires the system to build and navigate a mental model of the entire document's logical structure.

Defining the Core Hackathon Objective

Based on this comprehensive analysis, the core objective for the Hackrx 6.0 hackathon is clearly defined: to architect and build a robust question-answering system that can ingest complex, multi-page insurance policy documents and answer specific, natural language user queries with high factual accuracy. The system's output must be grounded exclusively in the provided documents and must include verifiable citations—specifically the source document name, section, and page number—for every piece of information used to construct its answer. This focus on accuracy and explainability will be the key differentiator and the primary measure of success.

II. Architectural Blueprint: A Multi-Stage Retrieval-Augmented Generation (RAG) Framework


Why RAG is the Optimal Architecture

To meet the stringent requirements of accuracy and explainability, the optimal architectural choice is Retrieval-Augmented Generation (RAG). RAG is a state-of-the-art AI framework designed specifically for knowledge-intensive tasks where factual grounding is non-negotiable.2 The core principle of RAG is to separate the roles of knowledge storage and language generation. Instead of relying on the vast, static, and sometimes inaccurate information baked into a Large Language Model (LLM) during its training, a RAG system forces the LLM to base its answers on a small, specific, and trusted set of context documents retrieved in real-time.5 For this hackathon, the insurance policies themselves constitute this trusted knowledge base.
This approach directly mitigates the primary risk associated with LLMs: "hallucination," or the generation of plausible but factually incorrect information.3 In the context of insurance, a hallucinated answer is not just an error; it's a critical failure. RAG provides a robust defense against this by constraining the LLM to the provided textual evidence.
Simpler alternatives, such as fine-tuning an LLM on a corpus of insurance documents, are ill-suited for this task. Fine-tuning is resource-intensive, expensive, and creates a static model; it cannot adapt to a new policy document without being retrained.7 More importantly, while fine-tuning can teach a model the
style and vocabulary of insurance language, it does not guarantee factual recall for specific clauses within a specific document, making it fundamentally less reliable for this use case.7 RAG, by its design, is dynamic, cost-effective, and built for verifiable accuracy.

High-Level System Diagram

The proposed architecture is a multi-stage pipeline divided into two distinct phases: an offline Indexing Pipeline that processes and prepares the insurance documents, and an online Inference Pipeline that handles user queries in real-time.
Figure 1: High-Level System Architecture

Code snippet


graph TD
    subgraph Offline Indexing Pipeline
        A --> B(High-Fidelity Parsing & Cleaning);
        B --> C{Intelligent Structural Chunking};
        C --> D;
        D --> E((Hybrid Index: Vector + Keyword));
    end

    subgraph Online Inference Pipeline
        F[User Query] --> G{Query Rewriting};
        G --> H(Hybrid Retrieval);
        E --> H;
        H --> I(Cross-Encoder Re-ranking);
        I --> J{Context Compression & Augmentation};
        J --> K[Grounded Prompt Generation];
        K --> L{LLM};
        L --> M;
    end



Component Responsibilities

Offline Indexing Pipeline: This is a one-time, preparatory process performed for each new insurance policy.
High-Fidelity Parsing & Cleaning: Ingests raw PDF documents and extracts text, tables, and structural metadata with maximum accuracy, removing artifacts like headers and footers.
Intelligent Structural Chunking: Divides the extracted content into semantically meaningful chunks based on the document's logical structure (sections, subsections, tables), not arbitrary length.
Text & Table Embedding Generation: Uses a text embedding model to convert each chunk into a numerical vector representation that captures its semantic meaning.
Hybrid Indexing: Stores the chunks and their vector representations in a specialized database that supports both semantic (vector) search and traditional keyword search.
Online Inference Pipeline: This process executes in real-time for every user question.
Query Rewriting: Analyzes the user's query and, if necessary, reformulates it or generates variations to improve retrieval performance.
Hybrid Retrieval: Searches the index using a combination of semantic and keyword matching to retrieve a broad set of potentially relevant document chunks.
Cross-Encoder Re-ranking: Employs a more powerful AI model to re-score and re-rank the initial set of retrieved chunks for maximum relevance to the specific user query.
Context Compression & Augmentation: Selects the top-ranked, most relevant chunks and formats them into a concise context block.
Grounded Prompt Generation: Constructs a detailed prompt that includes the user's question, the retrieved context, and strict instructions for the LLM on how to generate an answer.
LLM Response Generation: Sends the final prompt to an LLM, which synthesizes a natural language answer based only on the provided context.
Response with Citations: The final output delivered to the user, containing the generated answer and precise source citations for verification.
This architectural blueprint provides a clear and robust roadmap for developing a system that is not only functional but also accurate, reliable, and explainable—the key ingredients for success in this hackathon.

III. The Ingestion Pipeline: Transforming Policies into a Searchable Knowledge Base

The performance and reliability of the entire RAG system are fundamentally determined by the quality of its knowledge base. The Ingestion Pipeline is the factory that transforms raw, unstructured PDF documents into a highly structured, searchable, and context-aware index. Every step in this offline process is critical; a failure or poor choice at any stage will cascade through the system, irreparably harming the quality of the final output.

Step 1: High-Fidelity Document Parsing and Text Extraction

The initial challenge lies in the nature of the Portable Document Format (PDF) itself. A PDF is a format designed for visual presentation consistency, not for logical text flow. The order of text elements within the file's internal structure may not correspond to the reading order a human perceives, making naive text extraction highly unreliable.9 This can lead to jumbled sentences, broken paragraphs, and a complete loss of tabular structure, rendering the extracted content useless for downstream AI processing.
To overcome this, a sophisticated, hybrid extraction strategy is required. The selection of parsing tools is the first and most critical decision in the entire pipeline, as it dictates the fidelity of all subsequent steps.
Comparative Analysis of Python Libraries:
PyPDF2: While popular and easy to use, PyPDF2 is fundamentally inadequate for this task. It struggles with complex layouts, often failing to preserve text order, and provides no mechanism for reliable table extraction. Its use would introduce a high level of noise and data loss at the very first step, making it an unsuitable choice.11
Recommended Hybrid Strategy: A two-pronged approach is recommended to achieve the highest possible fidelity:
Primary Text Extraction with PyMuPDF (Fitz): PyMuPDF is the recommended library for primary text extraction due to its exceptional speed and, most importantly, its ability to extract not just the text but also rich metadata associated with it. This includes the exact coordinates (bounding boxes) of each word, font size, font weight (bold, italic), and color.9 This metadata is not just incidental; it is the raw material that enables the intelligent structural chunking in the next step.
Targeted Table Extraction with PDFPlumber: While PyMuPDF can extract text from tables, PDFPlumber is a specialized tool built on pdfminer.six that excels at identifying and parsing the structure of tables, even those without explicit grid lines.11 It can accurately reconstruct rows and columns, which is essential for preserving the relational data within the benefit schedules found in the sample policies.1
The implementation involves a two-pass process. First, PDFPlumber is used to identify the coordinates of all tables on a page and extract them as structured data (e.g., a list of lists). Then, PyMuPDF is used to extract all other text from the page, explicitly excluding the regions already identified as tables. This ensures that no data is processed twice and that each type of content is handled by the tool best suited for it.
The following table provides a clear justification for this hybrid parsing approach, highlighting the trade-offs and reasoning behind selecting specific tools for specific tasks.
Table 1: Comparison of PDF Parsing Libraries

Library
Speed
Layout/Metadata Fidelity
Table Extraction Accuracy
Ease of Use
Key Advantage
PyMuPDF (Fitz)
Very High
High (Coordinates, Font)
Low-Medium
Medium
Speed and rich metadata extraction.9
PDFPlumber
Medium
Medium
High
Medium
Superior table detection and extraction.11
PyPDF2
High
Very Low
Very Low
Easy
Simplicity for basic text dumps (not suitable here).11
Textract
Low
Low
Low
Easy
Handles multiple file types (DOCX, PNG), but is a wrapper around other tools.11
Camelot
Medium
N/A
Very High
Medium
Specialized for tables only; less flexible for mixed content.14


Step 2: Intelligent Chunking for Context Preservation

Once high-fidelity text and tables are extracted, the next critical step is to divide this content into "chunks" for embedding and retrieval. A naive approach, such as using LangChain's RecursiveCharacterTextSplitter with a fixed-size overlap, is a common but critical failure point for legal and insurance documents.17 This method will arbitrarily slice through the middle of a legal clause, separate a list from its introductory sentence, or break a table row in half, thereby destroying the semantic integrity and context of the information.
Structural Chunking Strategy: A far superior approach is structural chunking, which leverages the rich metadata extracted by PyMuPDF. By analyzing changes in font size, bolding, and indentation, the document's logical hierarchy can be programmatically inferred.
A larger, bold font indicates a chapter or major section header.1
A slightly smaller, bold font indicates a subsection.
Indented lists with bullet points or numbering represent sub-clauses.
The strategy is to define a set of rules based on these stylistic cues to segment the document. Each logical unit—be it a full paragraph, a complete nested list, or a subsection—becomes a single chunk. This ensures that semantically related sentences are kept together, providing the LLM with complete context during the generation phase.
Table Processing: Tables extracted by PDFPlumber require special handling. They should be converted from their list-of-lists format into a structured, human-readable, and LLM-friendly format like Markdown. For example:
Benefit
Percentage of Sum Insured
Accident Death (AD)
100%
Loss of toes - all
20%




This Markdown table is then treated as a single, self-contained chunk. This preserves the crucial relational information between columns and rows.


Metadata Enrichment: This is a non-negotiable step for ensuring explainability. Every single chunk created, whether text or table, must be enriched with a comprehensive metadata dictionary. This dictionary acts as the chunk's passport, carrying essential information about its origin. A sample metadata structure would be:
JSON
{
  "source_document": "CHOTGDP23004V012223.pdf",
  "page_number": 13,
  "section_header": "4. GENERAL EXCLUSIONS",
  "chunk_type": "text"
}

This metadata is not used during the semantic search itself but is carried along with the retrieved chunks. When the LLM generates its answer, it is instructed to pull from this metadata to create the citations, making the system's output verifiable and trustworthy.

Step 3: Vectorization and Embedding Strategy

Vectorization is the process of converting the text chunks into numerical representations, known as embeddings. An embedding model is a neural network that has been trained on a massive amount of text to understand the nuances of language. It maps each chunk of text to a high-dimensional vector in a way that chunks with similar semantic meanings are located close to each other in this vector space.5 This is what enables "semantic search"—finding documents based on meaning and intent, not just keyword matches.
Hackathon Model Recommendation: For the time-constrained environment of a hackathon, a high-performing, off-the-shelf sentence-transformer model is the ideal choice. The model sentence-transformers/all-mpnet-base-v2 is highly recommended. It provides an excellent balance of performance (i.e., the quality of the embeddings it produces) and efficiency, and it can be run locally on a standard machine or accessed via a simple API, avoiding complex setup.21
Expert Consideration for Production Systems: While all-mpnet-base-v2 is excellent for general purposes, a production-grade system for the legal or insurance domain would benefit significantly from a domain-specific model. Models like LEGAL-BERT have been pre-trained specifically on legal and contractual documents.21 This specialized training allows them to better grasp the subtle but critical differences in meaning between domain-specific terms (e.g., "shall" vs. "may," or "liability" vs. "indemnity"). Using such a model would result in a more nuanced and accurate vector space, leading to more precise retrieval. Mentioning this path for improvement demonstrates a deeper, forward-looking understanding of the problem space.

Step 4: Vector Storage and Indexing

The generated high-dimensional vectors must be stored in a specialized database capable of performing efficient similarity searches. A traditional relational database is not designed for this task; finding the "nearest neighbors" to a query vector in a high-dimensional space would require a computationally expensive full scan of the entire dataset. A vector database uses specialized indexing algorithms (like IVF or HNSW) to partition the vector space, allowing for near-instantaneous retrieval of the most similar vectors.23
Tool Selection: FAISS for Hackathon Agility:
FAISS (Facebook AI Similarity Search): For the hackathon, FAISS is the unequivocally correct choice. FAISS is a library, not a managed service. This means it is completely free, runs locally in-memory, and requires minimal setup.24 This eliminates dependencies on cloud services, network latency, and API keys, which are all potential points of failure and delay in a time-pressured environment. Its performance for the scale of a few dozen documents is exceptionally high, and its integration into the LangChain framework is mature and well-documented, allowing for rapid implementation.27
Pinecone and Managed Services: Cloud-native vector databases like Pinecone are the industry standard for production-scale applications. They offer benefits like serverless architecture, automatic scaling, real-time data updates, and advanced features like metadata filtering.25 However, these benefits come at the cost of financial expense and operational overhead (API key management, network configuration, potential data privacy concerns). These factors make such services an unnecessary complication for a self-contained hackathon project.
The following matrix justifies the selection of FAISS by tailoring the decision criteria to the specific constraints and goals of a hackathon.
Table 2: Vector Database Selection Matrix for Hackathon Context

Feature
FAISS
Pinecone
Type
Library 25
Managed Service (SaaS) 29
Cost Model
Free (Open Source) 25
Subscription-based (Usage/Time) 29
Setup Time
Minimal (pip install) 26
Moderate (Account setup, API key, index provisioning)
Scalability
Limited by local RAM
Highly Scalable (Cloud-native) 25
Performance (Local)
Extremely High (In-memory)
N/A (Network latency)
Development Overhead
Low
Moderate (API integration, network handling)
Hackathon Recommendation
Ideal
Overkill


IV. The Retrieval Engine: Achieving Unparalleled Precision and Recall

The retrieval engine is the heart of the RAG system. Its sole purpose is to find and return the most relevant document chunks in response to a user's query. The quality of the retrieved context directly determines the quality of the LLM's generated answer. A naive retrieval strategy, while simple to implement, is brittle and often fails to handle the nuances of complex queries, leading to inaccurate or incomplete answers. This section details a multi-stage retrieval architecture designed for maximum precision and recall.

The Weakness of Naive Vector Search

A simple vector search retrieves chunks based on semantic similarity alone. While powerful, this approach has predictable failure modes. For instance, a user query like "What is the policy number for the Cholamandalam travel insurance?" contains a specific term ("CHOTGDP23004V012223") that may have low semantic similarity to the rest of the document, causing it to be missed by a pure vector search. Conversely, a conceptual query like "What happens if my trip is canceled?" might retrieve chunks related to "trip interruption," as the two concepts are semantically close, even though they are distinct coverages with different conditions under the policy.1 This demonstrates the need for a more robust retrieval strategy that can handle both specific keywords and broad concepts.

Advanced Retrieval I: Implementing Hybrid Search

Hybrid search addresses the limitations of pure vector search by combining two distinct retrieval methods: dense (semantic) and sparse (keyword-based) search. This fusion ensures that the system benefits from the conceptual understanding of vector search while retaining the literal precision of keyword matching.30
Concept:
Dense Retrieval (Vector Search): This is the semantic search performed using the vector embeddings stored in FAISS. It excels at understanding user intent and finding conceptually related content, even if the exact wording differs.
Sparse Retrieval (Keyword Search): This is a traditional keyword search, most effectively implemented using the BM25 algorithm. BM25 is a ranking function that scores documents based on the frequency and inverse document frequency of the query terms, making it highly effective at finding documents containing specific, literal keywords.31
Implementation: The hybrid search process is executed in parallel:
Dual Indexing: During the ingestion pipeline, all text chunks are indexed in two separate systems: the FAISS index for their dense vector embeddings and a BM25 index (e.g., using the rank_bm25 Python library) for their raw text.
Parallel Querying: When a user query is received, it is sent to both the FAISS index (after being converted to a vector) and the BM25 index simultaneously. Each system returns its own ranked list of relevant chunks.
Result Fusion with RRF: The two separate ranked lists must be merged into a single, superior list. The recommended method for this is Reciprocal Rank Fusion (RRF). RRF is a simple yet highly effective algorithm that re-ranks documents based on their position in each input list, rather than their raw scores. This approach is powerful because it does not require score normalization between the two different systems. The final score for each document is calculated by summing the reciprocal of its rank in each list, and the documents are then re-sorted based on this new combined score.32

Advanced Retrieval II: The Re-ranking Layer for Ultimate Precision

While hybrid search significantly improves the quality of the retrieved candidates, a final re-ranking step can provide an additional, decisive boost in precision. This stage introduces a more powerful, computationally expensive model to scrutinize the top candidates from the hybrid search and produce the final, definitive ranking.34
Concept: The retrieval process up to this point has used bi-encoder models. In a bi-encoder setup, the query and the documents are encoded into vectors independently. The similarity search then happens in the vector space. This is very fast and scalable. A cross-encoder, by contrast, takes the user's query and a candidate document together as a single input. This allows the model to perform deep, bidirectional attention across both texts simultaneously, capturing much finer-grained nuances of relevance that a bi-encoder might miss.35 This process is much slower, which is why it is only applied to a small set of the most promising candidates (e.g., the top 20-25 results) returned by the hybrid search. This two-stage architecture—a fast but broad initial retrieval followed by a slow but highly accurate re-ranking—is a classic and powerful pattern for balancing speed and precision in modern search systems.
Implementation: The implementation involves adding a final step to the retrieval pipeline. After the RRF algorithm produces a single ranked list from the hybrid search, the top N candidates from this list are passed to a pre-trained cross-encoder model (e.g., a model from the sentence-transformers library). The cross-encoder outputs a new relevance score for each query-document pair. These scores are then used to create the final ranking. The top 3-5 chunks from this final list are then selected to be passed as context to the LLM.
The following table visually justifies the progressive complexity of the proposed retrieval pipeline, demonstrating how each stage addresses the weaknesses of the previous one.
Table 3: Retrieval Strategy Effectiveness

Retrieval Strategy
Example Query Type
Precision
Recall
Speed
Key Benefit
Keyword Search (BM25)
Specific Term ("Subrogation")
High
Low
Very High
Finds exact matches and rare terms.
Vector Search
Conceptual ("What if I get sick?")
Medium
High
High
Understands user intent and synonyms.
Hybrid Search (RRF)
Ambiguous ("Air ambulance cost")
High
High
High
Balances keyword precision with semantic understanding.31
Hybrid + Re-ranking
All Types
Very High
High
Medium
Achieves maximum relevance by deep analysis of top candidates.34


V. The Generation Core: Ensuring Factual, Explainable, and Context-Aware Responses

The final stage of the RAG pipeline is generation. Here, the meticulously retrieved and ranked context is passed to a Large Language Model (LLM) to synthesize a final, human-readable answer. The primary goal in this stage is to strictly control the LLM's behavior, ensuring its output is factually grounded, trustworthy, and directly answers the user's question.

The LLM as a Synthesizer, Not a Knower

It is crucial to reinforce the architectural principle that the LLM's role is not to access its own vast, pre-trained knowledge base. Its function is solely to act as a sophisticated reasoning and summarization engine that operates exclusively on the context provided in the prompt.7 This "grounding" is the cornerstone of the RAG approach and is what makes the system's outputs reliable and verifiable. The prompt must be engineered to enforce this constraint explicitly.

Prompt Engineering for Explainable AI (XAI)

Prompt engineering is the art and science of crafting inputs that guide an LLM to produce a desired output.7 For this application, the prompt is the primary mechanism for ensuring accuracy and enabling explainability. A robust, structured prompt template is required to instruct the LLM on its precise task, its constraints, and its required output format.37
The following template is recommended. It uses a system-level instruction to define the AI's persona and overall task, clearly separates the provided context from the user's question, and provides explicit, numbered instructions on how to behave.
Recommended Prompt Template:
<SYSTEM>
You are an expert AI assistant specialized in analyzing insurance policy documents. Your task is to answer the user's question based *only* on the provided context. Do not use any external knowledge or make assumptions.

**Instructions:**
1.  Carefully read the user's question and the provided context documents. The context is a list of chunks from the policy, each with metadata indicating its source document and page number.
2.  Formulate a clear, concise, and direct answer using only the information present in the context.
3.  For every statement or piece of information in your answer, you MUST provide a citation in the format ``. A single sentence may require multiple citations if it synthesizes information from different sources.
4.  If the information required to answer the question is not available in the provided context, you MUST respond with: "I'm sorry, but the information required to answer your question is not available in the provided documents." Do not attempt to answer from memory.
</SYSTEM>

<CONTEXT>
{retrieved_documents_with_metadata}
</CONTEXT>

<USER>
{user_question}
</USER>


This prompt structure achieves several key objectives:
Role-Playing: It sets the LLM's persona as a specialized expert.
Grounding: It repeatedly emphasizes the constraint to use only the provided context.
Forced Citation: It makes citation a mandatory part of the output format, which is the core of the system's explainability.
Graceful Failure: It provides a clear, safe fallback response when the necessary information has not been retrieved, preventing the LLM from inventing an answer.

Managing Conversational Context

To create a more natural, chatbot-like experience where users can ask follow-up questions, the system needs a basic form of memory. While complex state management systems are possible, a simple and effective strategy suitable for a hackathon is to append the history of the last few conversational turns to the new user query before the retrieval step.39
For example, if the conversation is:
User (Turn 1): "What is the coverage for air ambulance?"
AI (Turn 1): "The policy covers air ambulance services up to a maximum distance of 150 kms..."
User (Turn 2): "What if the distance is longer?"
Before retrieval for Turn 2, the system would formulate a new, context-enriched query like: "Previous question was about air ambulance coverage. New question is: What if the distance is longer?". This combined query is then sent to the retrieval engine. This provides sufficient context for the retriever to find relevant chunks about proportionate payments for distances over 150 kms, allowing the LLM to answer the follow-up question accurately.1 Limiting the history to the last 2-3 turns prevents the context from becoming too long and noisy.

VI. Implementation and Deployment Strategy for Hackathon Success

A brilliant architecture is only valuable in a hackathon if it can be implemented, demonstrated, and iterated upon within the severe time constraints. The choice of frameworks and tools is therefore a strategic decision aimed at maximizing development velocity and minimizing unnecessary complexity. The recommended technology stack—LangChain, FastAPI, and Streamlit—is specifically chosen to enable rapid, robust development of the entire RAG pipeline.

Framework Selection: LangChain for Rapid Orchestration

LangChain is an open-source framework designed to simplify the development of applications powered by LLMs. It provides a comprehensive suite of modular components, abstractions, and "chains" that encapsulate the logic for common RAG patterns.26 Adopting LangChain is a significant competitive advantage, as it allows the team to focus on the novel aspects of the architecture (like the hybrid retrieval and re-ranking logic) rather than re-implementing boilerplate code for data loading, chunking, and vector store interaction.
Key LangChain components that will be leveraged include:
Document Loaders: While a custom parsing solution is recommended, LangChain's loaders can serve as a quick starting point.
Text Splitters: Although structural chunking is the goal, LangChain's splitters can be used for initial prototyping.
Vector Store Integrations: LangChain offers a seamless, well-documented integration with FAISS, handling the complexities of adding documents, creating an index, and performing similarity searches with just a few lines of code.27
Retrievers: The framework provides a Retriever abstraction that simplifies the process of fetching documents, which can be customized to implement the hybrid search and re-ranking logic.
Prompt Templates: LangChain's PromptTemplate class is ideal for implementing the structured prompt required for the generation core.
By using LangChain, the team can assemble the end-to-end pipeline quickly, ensuring a functional prototype is available early in the hackathon for testing and refinement.26

Serving the Model: FastAPI for a High-Performance API

The backend of the application, which will expose the RAG pipeline as a service, should be built using FastAPI. FastAPI is a modern, high-performance Python web framework that is exceptionally well-suited for this task.41
Its key advantages include:
Performance: FastAPI is built on Starlette and Pydantic, making it one of the fastest Python frameworks available. Its native support for asynchronous operations is particularly beneficial, as it can handle I/O-bound tasks—like making API calls to an LLM or querying the vector database—very efficiently without blocking the server.
Ease of Use: The framework is designed to be intuitive and requires minimal boilerplate code to get a robust API up and running.
Automatic Interactive Documentation: A standout feature is its automatic generation of interactive API documentation (via Swagger UI and ReDoc). This provides a web-based interface where the team can directly test the API endpoints, upload files, and send queries without needing to write a separate client or use command-line tools like curl. This is invaluable for rapid debugging, iteration, and demonstration during a hackathon.42
The API will be structured with two primary endpoints:
POST /upload: An endpoint that accepts a PDF file, triggers the offline Ingestion Pipeline to process and index the document, and returns a success status.
POST /query: An endpoint that accepts a user's question (and optionally, a chat history), executes the online Inference Pipeline, and returns a JSON object containing the generated answer and a list of source citations.

User Interface: Streamlit for a Quick and Effective Demo

For the user-facing component, building a complex front-end with frameworks like React or Vue is a time-consuming task that is not a core part of the hackathon's AI challenge. Streamlit is the ideal solution. It is an open-source Python library that allows developers to create and share beautiful, custom web apps for machine learning and data science projects with surprisingly little code.26
Using Streamlit, the team can create a polished and interactive chat interface in a matter of hours, using only Python. The application would feature:
A file uploader widget to allow judges or users to upload a new insurance policy PDF.
A text input box for the user to type their questions.
A chat display area that shows the conversation history, with the AI's answers and their corresponding source citations clearly displayed.
This approach allows the team to produce a professional-looking and fully functional demo application that effectively showcases the power of the backend RAG pipeline, without getting bogged down in front-end web development.

VII. Performance Optimization and Future Enhancements

While building a functional RAG pipeline is the primary goal, incorporating key performance optimizations and articulating a vision for future enhancements can significantly elevate the project during final judging. This section outlines critical optimizations for the hackathon and potential next steps for a production-grade system.

Hackathon-Critical Optimizations

Within the time constraints of a hackathon, optimizations should focus on improving the user's perceived performance and the efficiency of the most expensive parts of the pipeline.
Perceived Latency with Streaming: The single most impactful optimization for user experience is to stream the LLM's response. The generation step is typically the slowest part of the inference pipeline. Instead of waiting for the entire answer to be generated before displaying it, streaming sends the response back to the user token-by-token as it is generated by the LLM.45 This dramatically reduces the "time-to-first-token," making the application feel almost instantaneous, even if the full response takes several seconds to complete. Both OpenAI's API and FastAPI support streaming responses, making this a feasible and high-impact feature to implement.
Token Efficiency through Context Compression: The advanced retrieval techniques of re-ranking and hybrid search are not just for improving accuracy; they are also critical performance optimizations.46 The cost (both in terms of latency and potential API fees) of an LLM call is directly proportional to the number of input and output tokens. By using a sophisticated retrieval pipeline to select a small number of highly relevant chunks (context compression), we significantly reduce the number of tokens sent to the LLM.47 This leads to faster generation times and lower operational costs, a key consideration for any scalable AI application. The prompt should explicitly instruct the model to be concise, further reducing the number of generated output tokens and thereby decreasing latency.45

Avenues for Improvement (Post-Hackathon Vision)

Articulating a clear and credible roadmap for future development demonstrates strategic thinking beyond the immediate hackathon prototype.
Domain-Specific Fine-Tuning: The next logical step to improve retrieval accuracy would be to fine-tune the embedding model. By training a model like bge-base-en-v1.5 on a large, curated corpus of insurance and legal documents, the model would develop a more nuanced understanding of the domain's specific semantics. This would lead to a higher-quality vector space and more precise retrieval of relevant context, directly improving the quality of the final answers.
Advanced RAG Techniques: The field of RAG is evolving rapidly. Future enhancements could incorporate more advanced techniques that add layers of reasoning and self-correction to the pipeline. For example, SELF-RAG introduces a "reflection" step where the model critiques its own retrieved documents and generated answers for relevance and factual support.49 Similarly,
Corrective RAG (CRAG) adds a retrieval evaluator that can trigger web searches to augment or correct the information found in the static document base if it is deemed insufficient or outdated.49 These techniques represent a move towards more autonomous and robust reasoning systems.
Evolution to an Agentic Architecture: The current Q&A system can be envisioned as a single tool in a more powerful, multi-tool AI agent. This agent could be tasked with more complex, multi-step tasks. For example, a user could ask, "Compare the air ambulance coverage in the Edelweiss policy with the Bajaj Allianz policy." The agent would first use the RAG tool twice—once for each policy—to extract the relevant clauses. Then, it would use a "comparison" tool (another LLM call with a specific prompt) to analyze the two retrieved contexts and generate a structured summary of the differences. This agentic approach represents a significant leap in capability, moving from simple information retrieval to complex analysis and synthesis across multiple documents.

Conclusions

The challenge of accurately interpreting complex insurance policies demands a solution that is robust, reliable, and transparent. The proposed multi-stage Retrieval-Augmented Generation (RAG) architecture provides a comprehensive blueprint for building such a system. By prioritizing high-fidelity data ingestion, employing a sophisticated hybrid retrieval and re-ranking engine, and enforcing factual grounding through meticulous prompt engineering, this architecture directly addresses the core complexities of the problem domain.
The recommended technology stack—leveraging PyMuPDF and PDFPlumber for parsing, FAISS for agile vector storage, LangChain for rapid pipeline orchestration, FastAPI for a high-performance backend, and Streamlit for a quick-to-deploy user interface—is strategically tailored for success within the demanding constraints of a hackathon. This approach maximizes development velocity while implementing state-of-the-art techniques.
Ultimately, this report outlines not just a functional prototype, but a pathway to a best-in-class solution. The emphasis on verifiable citations and advanced retrieval methods provides a distinct competitive advantage, resulting in a system that is not only intelligent but also trustworthy. The outlined future enhancements, from domain-specific model fine-tuning to a full agentic architecture, demonstrate a clear vision for evolving this hackathon project into a powerful, production-ready enterprise AI application.
Works cited
BAJHLIP23020V012223.pdf
What is Retrieval-Augmented Generation (RAG)? - Google Cloud, accessed on August 5, 2025, https://cloud.google.com/use-cases/retrieval-augmented-generation
RAG Tutorial: A Beginner's Guide to Retrieval Augmented Generation - SingleStore, accessed on August 5, 2025, https://www.singlestore.com/blog/a-guide-to-retrieval-augmented-generation-rag/
What Is Retrieval-Augmented Generation aka RAG - NVIDIA Blog, accessed on August 5, 2025, https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/
Semantic Search and RAG: Key Differences and Use Cases - Signity Solutions, accessed on August 5, 2025, https://www.signitysolutions.com/blog/semantic-search-and-rag
Retrieval Augmented Generation (RAG) Question Answering - What's deepset AI Platform?, accessed on August 5, 2025, https://docs.cloud.deepset.ai/docs/generative-question-answering
RAG vs fine-tuning vs. prompt engineering - IBM, accessed on August 5, 2025, https://www.ibm.com/think/topics/rag-vs-fine-tuning-vs-prompt-engineering
Choosing the Right AI Technique: Prompt Engineering vs. RAG vs. Fine-Tuning - Medium, accessed on August 5, 2025, https://medium.com/@nitishsingh.imnu/choosing-the-right-ai-technique-prompt-engineering-vs-rag-vs-fine-tuning-a61f16a9fc52
Document Intelligence: The art of PDF information extraction, accessed on August 5, 2025, https://www.statcan.gc.ca/en/data-science/network/pdf-extraction
How can I extract tables as structured data from PDF documents? - Stack Overflow, accessed on August 5, 2025, https://stackoverflow.com/questions/17591426/how-can-i-extract-tables-as-structured-data-from-pdf-documents
Top Python libraries for text extraction from PDFs - AZ Big Media, accessed on August 5, 2025, https://azbigmedia.com/business/top-python-libraries-for-text-extraction-from-pdfs/
Python Packages for PDF Data Extraction | by Rucha Sawarkar | Analytics Vidhya | Medium, accessed on August 5, 2025, https://medium.com/analytics-vidhya/python-packages-for-pdf-data-extraction-d14ec30f0ad0
[D] Choosing a pdf processing package in Python : r/MachineLearning - Reddit, accessed on August 5, 2025, https://www.reddit.com/r/MachineLearning/comments/191uyuq/d_choosing_a_pdf_processing_package_in_python/
A Comparison of python libraries for PDF Data Extraction for text, images and tables, accessed on August 5, 2025, https://pradeepundefned.medium.com/a-comparison-of-python-libraries-for-pdf-data-extraction-for-text-images-and-tables-c75e5dbcfef8
jsvine/pdfplumber: Plumb a PDF for detailed information about each char, rectangle, line, et cetera — and easily extract text and tables. - GitHub, accessed on August 5, 2025, https://github.com/jsvine/pdfplumber
Python -- Parsing files (docx, pdf and odt) and converting the content into my data model, accessed on August 5, 2025, https://stackoverflow.com/questions/24860635/python-parsing-files-docx-pdf-and-odt-and-converting-the-content-into-my-d
Basic RAG - Mistral AI Documentation, accessed on August 5, 2025, https://docs.mistral.ai/guides/rag/
Master RAG Optimization: Key Strategies for AI Engineers - Galileo AI, accessed on August 5, 2025, https://galileo.ai/blog/rag-performance-optimization
Hakim: Farsi Text Embedding Model - arXiv, accessed on August 5, 2025, https://arxiv.org/html/2505.08435v1
Getting started with Amazon Titan Text Embeddings in Amazon Bedrock - AWS, accessed on August 5, 2025, https://aws.amazon.com/blogs/machine-learning/getting-started-with-amazon-titan-text-embeddings/
What types of embedding models are best for legal documents? - Milvus, accessed on August 5, 2025, https://milvus.io/ai-quick-reference/what-types-of-embedding-models-are-best-for-legal-documents
Enhancing Legal Research with Domain-Adapted Semantic Search | Free Law Project, accessed on August 5, 2025, https://free.law/2025/03/11/semantic-search
Faiss vs. Pinecone: Ideal Vector Database Comparison - MyScale, accessed on August 5, 2025, https://myscale.com/blog/faiss-vs-pinecone-ideal-vector-database/
Comparing Pinecone, Chroma DB and FAISS: Exploring Vector Databases, accessed on August 5, 2025, https://community.hpe.com/t5/insight-remote-support/comparing-pinecone-chroma-db-and-faiss-exploring-vector/td-p/7210879
A Beginner's Guide to Vector Databases: Pinecone, FAISS & Chroma Explained - Medium, accessed on August 5, 2025, https://medium.com/@rohanmistry231/a-beginners-guide-to-vector-databases-pinecone-faiss-chroma-explained-d4eb3840f7c8
Building RAG application using Langchain , OpenAI , FAISS | by Kishore B | Medium, accessed on August 5, 2025, https://medium.com/@Kishore-B/building-rag-application-using-langchain-openai-faiss-3b2af23d98ba
Faiss | 🦜️ LangChain, accessed on August 5, 2025, https://python.langchain.com/docs/integrations/vectorstores/faiss/
Build RAG Chatbot with LangChain, Faiss, NVIDA Llama 3 70B Instruct, and voyage-3 - Zilliz, accessed on August 5, 2025, https://zilliz.com/tutorials/rag/langchain-and-faiss-and-nvida-llama-3-70b-instruct-and-voyage-3
Faiss vs Pinecone: Comparing Vector Search Databases - Scout, accessed on August 5, 2025, https://www.scoutos.com/blog/faiss-vs-pinecone-comparing-vector-search-databases
RAG using Hybrid Search with Milvus and LlamaIndex, accessed on August 5, 2025, https://milvus.io/docs/llamaindex_milvus_hybrid_search.md
Hybrid Search: Vector + Keyword Techniques for better RAG retrieval, accessed on August 5, 2025, https://www.machinelearningplus.com/gen-ai/hybrid-search-vector-keyword-techniques-for-better-rag/
Optimizing RAG with Hybrid Search & Reranking | VectorHub by Superlinked, accessed on August 5, 2025, https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking
Hybrid Search: Combining Semantic and Keyword Approaches for Enhanced Information Retrieval | by Pia Riachi | Google Cloud - Medium, accessed on August 5, 2025, https://medium.com/google-cloud/hybrid-search-combining-semantic-and-keyword-approaches-for-enhanced-information-retrieval-6a7c046c89ea
Reranking Explained: Why It Matters for RAG Systems - Chatbase, accessed on August 5, 2025, https://www.chatbase.co/blog/reranking
Re-ranking in Retrieval Augmented Generation: How to Use Re-rankers in RAG - Chitika, accessed on August 5, 2025, https://www.chitika.com/re-ranking-in-retrieval-augmented-generation-how-to-use-re-rankers-in-rag/
Enhancing RAG Pipelines with Re-Ranking | NVIDIA Technical Blog, accessed on August 5, 2025, https://developer.nvidia.com/blog/enhancing-rag-pipelines-with-re-ranking/
Prompt Engineering Patterns for Successful RAG Implementations - DEV Community, accessed on August 5, 2025, https://dev.to/shittu_olumide_/prompt-engineering-patterns-for-successful-rag-implementations-2m2e
Explainable AI: A Retrieval-Augmented Generation Based Framework for Model Interpretability - SciTePress, accessed on August 5, 2025, https://www.scitepress.org/Papers/2025/132413/132413.pdf
Build a Retrieval Augmented Generation (RAG) App: Part 2 | 🦜️ LangChain, accessed on August 5, 2025, https://python.langchain.com/docs/tutorials/qa_chat_history/
Open-source enhancements to LangChain PostgreSQL | Google Cloud Blog, accessed on August 5, 2025, https://cloud.google.com/blog/products/ai-machine-learning/open-source-enhancements-to-langchain-postgresql
Building a Retrieval-Augmented Generation (RAG) API and Frontend with FastAPI and React Native - DEV Community, accessed on August 5, 2025, https://dev.to/vivekyadav200988/building-a-retrieval-augmented-generation-rag-api-and-frontend-with-fastapi-and-react-native-2n7k
Python RAG API Tutorial with LangChain & FastAPI – Complete Guide - Vitalii Honchar, accessed on August 5, 2025, https://vitaliihonchar.com/insights/python-rag-api
Building a RAG Pipeline with FastAPI, Haystack, and ChromaDB for URLs in Python, accessed on August 5, 2025, https://www.aihello.com/resources/blog/building-a-rag-pipeline-with-fastapi-haystack-and-chromadb-for-urls-in-python/
Create Simple RAG based AI Chat bot | Python+LangChain+FAISS Vector Database(No API Keys Needed!) - YouTube, accessed on August 5, 2025, https://www.youtube.com/watch?v=hmqYxByTlRs
Latency optimization - OpenAI API, accessed on August 5, 2025, https://platform.openai.com/docs/guides/latency-optimization
Optimizing RAG pipelines for developers: Tips, tools, and techniques | We Love Open Source, accessed on August 5, 2025, https://allthingsopen.org/articles/optimizing-rag-pipelines-developers-tips-tools-techniques
Contextual compression for RAG based applications. - GitHub, accessed on August 5, 2025, https://github.com/SrGrace/Contextual-Compression
Contextual Compression: LangChain | LlamaIndex | by Sourav Verma - Medium, accessed on August 5, 2025, https://medium.com/@SrGrace_/contextual-compression-langchain-llamaindex-7675c8d1f9eb
Advanced RAG Techniques - Pinecone, accessed on August 5, 2025, https://www.pinecone.io/learn/advanced-rag-techniques/
