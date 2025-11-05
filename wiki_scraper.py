import wikipedia


def get_wikipedia_pages(query, limit=5):
    ##Searches query on Wikipedia and returns the search results.

    try:
        pages = wikipedia.search(query, results=limit)
    except Exception as e:
        return f"An error occurred while searching: {e}"
    return pages


def fetch_wikipedia_page(pages):
    ##Fetches the content of the pages from topic list
    wiki_content = []
    for title in pages:
        try:
            page = wikipedia.page(title)
            wiki_content.append({"title": page.title, "content": page.content})
        except wikipedia.exceptions.PageError as e:
            return f"Skipping Missing Page {e}"
            continue
        except wikipedia.exceptions.DisambiguationError as e:
            return f"Disambiguation Error: {e}"
            continue
        except Exception as e:
            return f"An error occurred: {e}"
            continue
    return wiki_content


def scraper_bot(query, limit=5):
    pages = get_wikipedia_pages(query, limit)
    if isinstance(pages, str):
        return pages  # Return error message if any
    wiki_content = fetch_wikipedia_page(pages)
    return wiki_content
