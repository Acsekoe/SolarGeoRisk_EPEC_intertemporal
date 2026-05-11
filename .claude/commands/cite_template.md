# Command to cite a scientific paper

Your goal is to find, verify, and add a proper citation for a specified concept, claim, or sentence in the paper.

## Setup: API Keys

This command uses two APIs. Replace the placeholders below with your own keys before use.

- **Semantic Scholar** (free, request at https://www.semanticscholar.org/product/api): `<SEMANTIC_SCHOLAR_API_KEY>`
- **Elsevier / Scopus** (free for academics at https://dev.elsevier.com/): `<ELSEVIER_API_KEY>`

You can either substitute the placeholders inline, or export the keys as environment variables and reference them in the curl commands (e.g., `-H "x-api-key: $SEMANTIC_SCHOLAR_API_KEY"`).

## Search Protocol

### Step 1: Search via Semantic Scholar (Primary)

Use this exact curl template:

```bash
curl -s -H "x-api-key: <SEMANTIC_SCHOLAR_API_KEY>" \
  "https://api.semanticscholar.org/graph/v1/paper/search?query=QUERY_HERE&fields=title,authors,year,abstract,citationCount,journal,externalIds,isOpenAccess,openAccessPdf&limit=10"
```

**Query Construction:**
- Replace `QUERY_HERE` with search terms joined by `+` (e.g., `demand+response+flexibility+residential`)
- Keep queries focused: 3-5 key terms work best
- No need for field operators — the search is semantic, not structured

**Full Example — Search for LLM energy management papers:**

```bash
curl -s -H "x-api-key: <SEMANTIC_SCHOLAR_API_KEY>" \
  "https://api.semanticscholar.org/graph/v1/paper/search?query=large+language+model+energy+management&fields=title,authors,year,abstract,citationCount,journal,externalIds,isOpenAccess,openAccessPdf&limit=10"
```

### Step 2: Parse the JSON Response

Extract from each entry in `["data"]`:
- `title` — Paper title
- `authors` — List of author objects (each has `name`)
- `year` — Publication year
- `abstract` — Abstract text (may be `null` for some publishers)
- `citationCount` — Number of citations
- `journal.name` — Journal/conference name
- `journal.volume` — Volume number
- `journal.pages` — Page numbers
- `externalIds.DOI` — DOI
- `isOpenAccess` — Whether full text is freely available
- `openAccessPdf.url` — Direct link to PDF (if open access)

### Step 3: Verify Abstracts (MANDATORY)

Before selecting a paper, you MUST read its abstract to verify relevance.

**If abstract is available from search results:** Read it directly from the response.

**If abstract is `null` (publisher-blocked):** Fall back to the Scopus Abstract Retrieval API:

```bash
curl -s "https://api.elsevier.com/content/abstract/doi/{DOI}?view=META_ABS" \
  -H "X-ELS-APIKey: <ELSEVIER_API_KEY>" \
  -H "Accept: application/json" | jq '.["abstracts-retrieval-response"] | {title: .coredata["dc:title"], abstract: .coredata["dc:description"], journal: .coredata["prism:publicationName"]}'
```

**Important:**
- Replace `{DOI}` with the paper's DOI (e.g., `10.1016/j.enpol.2024.114094`)
- Always verify abstracts for the top 2-3 promising papers
- Read the abstract carefully to confirm the paper actually supports the claim
- Do NOT cite a paper based solely on its title — the abstract must confirm relevance

### Step 4: Explore Citation Chains (Optional but Recommended)

When you find a highly relevant paper, use the citations endpoint to discover recent follow-up work:

```bash
curl -s -H "x-api-key: <SEMANTIC_SCHOLAR_API_KEY>" \
  "https://api.semanticscholar.org/graph/v1/paper/DOI:{DOI}/citations?fields=title,year,citationCount,journal,externalIds&limit=10"
```

This returns papers that cite the given paper — useful for finding the latest work building on foundational studies.

### Step 5: Look Up a Specific Paper by DOI

If you already have a DOI and need its details:

```bash
curl -s -H "x-api-key: <SEMANTIC_SCHOLAR_API_KEY>" \
  "https://api.semanticscholar.org/graph/v1/paper/DOI:{DOI}?fields=title,authors,year,abstract,citationCount,journal,externalIds,isOpenAccess,openAccessPdf"
```

## Fallback: Scopus Search

If Semantic Scholar returns insufficient results (e.g., very niche Elsevier-specific topic), use Scopus:

```bash
curl -s "https://api.elsevier.com/content/search/scopus?query=QUERY_HERE" \
  -H "X-ELS-APIKey: <c316f7c01c3319753a2b184c4cf39ef8>" \
  -H "Accept: application/json"
```

**Scopus Query Templates:**

| Search Type | Template |
|-------------|----------|
| Keywords in title/abstract/keywords | `TITLE-ABS-KEY%28term1%20AND%20term2%29` |
| Exact phrase | `TITLE-ABS-KEY%28%22exact%20phrase%22%29` |
| By author | `AUTH%28LastName%29` |
| By journal | `SRCTITLE%28Journal%20Name%29` |
| Recent only | `TITLE-ABS-KEY%28term%29%20AND%20PUBYEAR%20%3E%202022` |
| Combined | `AUTH%28Smith%29%20AND%20SRCTITLE%28Applied%20Energy%29` |

Parse Scopus results from `["search-results"]["entry"]`:
- `dc:title`, `dc:creator`, `prism:publicationName`, `prism:coverDate`, `prism:doi`, `prism:volume`, `prism:pageRange`, `citedby-count`

## Selection Criteria

Prioritize papers that:
- Directly support the claim or concept (verified via abstract)
- Are recent (prefer 2023+, unless seminal work)
- Have high citation counts
- Match the technical level and context of the paper
- Have abstracts that explicitly mention the concept being cited

## Citation Addition Process

1. **Check for Existing Citation**: Search the .bib file first to avoid duplicates

2. **Get Complete Metadata**: If search results lack full author list, fetch from DOI:
   ```bash
   curl -s "https://api.crossref.org/works/DOI_HERE" | jq '.message | {title, author, published}'
   ```

3. **Generate Citation Key**: Use format `firstauthor_keyword_year`
   - First author's last name (lowercase)
   - Key descriptive word from title (lowercase)
   - Publication year (4 digits)
   - Example: `smith_flexibility_2026`

4. **Format BibTeX Entry**:
   - Use `@article` for journal papers
   - Use `@inproceedings` for conference papers
   - Use `@misc` for arXiv preprints or technical reports
   - Include: title, author, year, journal/booktitle, volume, pages, doi
   - For >5 authors: "FirstAuthor, SecondAuthor, ThirdAuthor and others"

5. **Add to .bib**: Insert the new entry in alphabetical order by citation key

6. **Cite in main.tex**: Add `\cite{citation_key}` at the appropriate location

## Output Format

Provide:
1. Brief justification for why this paper supports the claim
2. The complete BibTeX entry you're adding
3. Confirmation of where you added the citation in main.tex (line number and context)

## Error Handling

- If Semantic Scholar returns no results, try Scopus as fallback
- If both return no results, try broader/different search terms
- If metadata incomplete, use CrossRef API on the DOI
- If abstract unavailable from both sources, note this and proceed with caution
- If claim is too broad, ask user to specify the exact concept to cite
